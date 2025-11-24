from function_lib import *

# ----------------- config -----------------
n_instance   = 1000
input_dim    = 5
noise        = 0.5
deg          = 2
grid_width   = 5
epoch        = 100
learning_rate = 0.001
n_simulation = 50

model_type   = "mse"     # choose between "mse" and "spo"
ModelClass   = get_model_class(model_type)

# ----------------- paths ------------------
base_path = change_to_py_file_dir(True)
base      = os.path.join(base_path, "Data", "Shortest_path")

ckpt_root = Path(base_path) / "Result" / "Train" / "ckpt_dir"
log_dir  = Path(base_path) / "Result" / "Train" / "CSVlogger"
ckpt_root.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

# ----------------- graph / solver ---------
G      = define_graph(grid_width)
solver = ShortestPathSolver(G)

# ----------------- results ----------------
total_result  = {}
regret_result = []

# =========================================================
# main loop over seeds
# =========================================================
for seed in range(n_simulation):
    # ---- reproducibility ----
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- data ----
    y_train, x_test, y_test, train_dl, valid_dl, test_dl = load_shortest_path_setting_with_splits(base, n_instance, input_dim, deg, noise, grid_width, seed, batch_size=32)

    # optional: pre-compute solpool if you actually use it somewhere
    # otherwise you can safely delete these two lines
    # y_train_t = torch.from_numpy(y_train).float()
    # solpool   = batch_solve(solver, y_train_t, relaxation=False)

    ckpt_dir = ckpt_root / f"{model_type}_seed_{seed}"

    # clean checkpoint directory for this seed
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- network (new net for each seed!) ----
    net = nn.Linear(input_dim, 2 * grid_width * (grid_width - 1))

    # ---- callbacks (recreated each seed) ----
    checkpoint_callback = ModelCheckpoint(
        monitor="val_regret",                 # must match your logged metric name
        dirpath=str(ckpt_dir),
        filename="model-{epoch:02d}-{val_regret:.4f}",
        mode="min",
        save_top_k=1
    )

    early_stop_callback = EarlyStopping(
        monitor="val_regret",
        mode="min",
        patience=10,
        min_delta=0.0,
        verbose=True
    )

    # ---- logger ----
    tb_logger = create_csv_logger(
        log_dir=str(log_dir),
        experiment_name=model_type,
        seed=seed
    )

    # ---- model ----
    model = ModelClass(
        net=net,
        solver=solver,
        lr=learning_rate,
        max_epochs=epoch
    )

    # ---- trainer ----
    trainer = pl.Trainer(
        max_epochs=epoch,
        min_epochs=5,
        accelerator="auto",            # "auto" is safer; uses GPU if available
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tb_logger,
        enable_progress_bar=True,
        deterministic=True
    )

    # optional sanity check: initial validation
    trainer.validate(model, dataloaders=valid_dl)

    # ---- training ----
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    # ---- load best model and test ----
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        raise RuntimeError(
            f"No checkpoint saved for seed {seed}. "
            "Check that 'val_regret' is logged in validation_step/validation_epoch_end."
        )

    best_model = ModelClass.load_from_checkpoint(
        checkpoint_path=best_model_path,
        net=nn.Linear(input_dim, 2 * grid_width * (grid_width - 1)),  # fresh net with same shape
        solver=solver,
        lr=learning_rate,
        max_epochs=epoch
    )

    test_result = trainer.test(best_model, dataloaders=test_dl)
    regret_result.append(test_result[0]["test_regret"])

total_result[model_type] = regret_result
print(f"{model_type} regrets over {n_simulation} seeds:", regret_result)