import os
import pickle
from typing import Optional, Union

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.AIQ import get_non_descreasing_convex_hull_of
from evaluation.utils import (model_metric_dict,
                              transform_batch_df_to_wide_format)
from routers.common import MODELS_TO_ROUTE_NAMES
from utils import generate_datetime_str, parse_model_parameters


class DefaultEvaluationResult:
    def __init__(
        self,
        router_name: str,
        results: pd.DataFrame,
        metrics: list,
        model_param_dict: dict = model_metric_dict,
    ):
        """
        Evaluation result object that stores the results of an evaluation run

        :param router_name: Name of the router
        :param results: Results dataframe, in long format
        :param metrics: AbstractMetric objects for computing metrics on the results
        :param model_param_dict: Dictionary of router base name and the parameters used for that router
        """
        self.router_name = router_name
        self.router_type = router_name.split("|")[0]
        self.long_results = results
        self.results = transform_batch_df_to_wide_format(results, model_param_dict)
        self.metrics = metrics
        self.model_param_dict = model_param_dict

    def save(self, path):
        with fsspec.open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with fsspec.open(path, "rb") as f:
            return pickle.load(f)

    def get_plotting_data(
        self,
        eval_name: Optional[str] = None,
        models_to_route: list[str] = (" ",),
    ):
        if eval_name is not None:
            df_to_plot = self.results[
                self.results["eval_name"].str.startswith(eval_name)
            ]
            df_to_plot = df_to_plot.set_index("eval_name")
            df_to_plot.loc["mean"] = df_to_plot.mean()
            df_to_plot = df_to_plot[df_to_plot.index == "mean"]
        else:
            # Get average accuracy for each model
            df_to_plot = self.results.set_index("eval_name")
            df_to_plot.loc["mean"] = df_to_plot.mean()
            df_to_plot = df_to_plot[df_to_plot.index == "mean"]
        model_columns = [col for col in df_to_plot.columns if self.router_type in col]
        llm_columns = [
            col
            for col in df_to_plot.columns
            if any(model in col for model in models_to_route + ["oracle"])
        ]

        # Preparing routing model data for plotting
        routing_model_data = {}
        for col in model_columns:
            # Split off the last part of the column name, which is the value_type
            col_names = col.split("|")
            # If cascading router has different setup
            if self.router_type == "cascading router":
                col_name = "|".join([col_names[0]] + col_names[2:4])
            else:
                col_name = col.split("|")[:-2]
                col_name = "|".join(col_name)
            if col_name not in routing_model_data:
                routing_model_data[col_name] = {}
            model_params = parse_model_parameters(col)
            # Max cost per response for cascading router is similar to willingness_to_pay
            if "max_cost_per_response" in model_params.keys():
                wtp_dict_key = float(model_params["max_cost_per_response"])
            else:
                wtp_dict_key = float(model_params["willingness_to_pay"])
            if wtp_dict_key not in routing_model_data[col_name]:
                routing_model_data[col_name][wtp_dict_key] = {}
            routing_model_data[col_name][wtp_dict_key][
                model_params["value_type"]
            ] = df_to_plot[col].values[0]

        # Adding LLM data as scatter points
        llm_data = {}
        for col in llm_columns:
            model_name, metric = col.split("|")
            value = df_to_plot[col].values[0]
            if "performance" in metric:
                accuracy = value
                total_cost = df_to_plot[f"{model_name}|total_cost"].iloc[0]
                llm_data[model_name] = (total_cost, accuracy)
        return routing_model_data, llm_data


class EvaluationResultCollection:
    def __init__(self, results: list[DefaultEvaluationResult]):
        self.evaluation_results = results

    def to_pickle(self, path):
        with fsspec.open(path, "wb") as f:
            pickle.dump(self, f)

    def to_csv(self, path):
        # combine all the evaluation results into one dataframe
        all_results = pd.concat(
            [result.long_results for result in self.evaluation_results]
        )
        all_results = all_results.drop_duplicates()
        all_results.to_csv(path, index=False)

    @staticmethod
    def load(path):
        with fsspec.open(path, "rb") as f:
            return pickle.load(f)

    def get_eval_names(self) -> list:
        return list(
            set(
                [
                    eval_name
                    for eval_result in self.evaluation_results
                    for eval_name in eval_result.results["eval_name"]
                ]
            )
        )

    def get_router_names(self) -> dict:
        # Get the unique names not in MODELS_TO_ROUTE or oracle in the results
        all_router_names = {}
        for eval_result in self.evaluation_results:
            model_columns = [
                col
                for col in eval_result.results.columns
                if eval_result.router_type in col
            ]
            if eval_result.router_type == "cascading router":
                # Cascading router has different setup
                model_columns = [
                    "|".join([col.split("|")[0]] + col.split("|")[2:-1])
                    for col in model_columns
                ]
            else:
                model_columns = ["|".join(col.split("|")[:-2]) for col in model_columns]
            all_router_names[eval_result.router_type] = list(set(model_columns))
        # Get unique ones, as total_cost, performance, etc can duplicate them
        return all_router_names

    def get_best_router_variant(
        self, eval_name: str, models_to_route: list[str] = (" ",)
    ):
        router_data = []
        if isinstance(models_to_route, tuple):
            models_to_route = list(models_to_route)
        for evaluation_result in self.evaluation_results:
            router_results, _ = evaluation_result.get_plotting_data(
                eval_name, models_to_route=models_to_route
            )
            router_data.append(router_results)
        # LLM data should be the same for all routers, so just take the first one
        # Pick the llm_data with the most data
        # Pick it per embedding to use
        router_data_best = []
        for router_results in router_data:
            embedding_model_best = {}
            best_name = []
            cost_performance = []
            for router_variant in router_results.keys():
                best_performance = 0.0
                lowest_cost = 1e10
                router_variant_results = router_results[router_variant]
                if "cascading" in router_variant:
                    continue
                # Get embedding model
                embedding_model = router_variant.split("|embedding:")[-1].split("|")[0]
                # Each variant has a different willingness_to_pay_value
                for willingness_to_pay_value in sorted(router_variant_results):
                    data = router_results[router_variant][willingness_to_pay_value]
                    if data["performance"] > best_performance and not np.isnan(
                        data["performance"]
                    ):
                        best_performance = data["performance"]
                        lowest_cost = data["total_cost"]
                best_name.append(router_variant)
                cost_performance.append((lowest_cost, best_performance))
                if embedding_model not in embedding_model_best:
                    embedding_model_best[embedding_model] = (
                        router_variant,
                        (lowest_cost, best_performance),
                    )
                else:
                    if best_performance > embedding_model_best[embedding_model][1][
                        1
                    ] or (
                        best_performance == embedding_model_best[embedding_model][1][1]
                        and lowest_cost < embedding_model_best[embedding_model][1][0]
                    ):
                        embedding_model_best[embedding_model] = (
                            router_variant,
                            (lowest_cost, best_performance),
                        )
            # Sort by best performance, then lowest cost
            # Get the indicies, to sort best_name and cost_performance at the same time
            idxes = sorted(
                range(len(cost_performance)),
                key=lambda x: (cost_performance[x][1], -cost_performance[x][0]),
                reverse=True,
            )
            best_name = [best_name[i] for i in idxes]
            cost_performance = [cost_performance[i] for i in idxes]
            # router_data_best.append({best_name[0]: router_results[best_name[0]]})
            for embedding_model, best_variant in embedding_model_best.items():
                print(
                    f"Best variant for {best_variant[0].split('|')[0]} {embedding_model} has max performance {best_variant[1][1]}"
                )
                router_data_best.append(
                    {best_variant[0]: router_results[best_variant[0]]}
                )
        return router_data_best

    def _get_performance_vs_cost_plot_data(
        self,
        eval_name: Optional[str] = None,
        router_names: list[str] = None,
        models_to_route: list[str] = (" ",),
    ):
        router_data = []
        llm_data = []
        for evaluation_result in self.evaluation_results:
            router_results, llm_results = evaluation_result.get_plotting_data(
                eval_name, models_to_route=models_to_route
            )
            if router_names is not None:
                # Only keep the keys that are in router_names
                router_results = {
                    router_name: router_results[router_name]
                    for router_name in router_results.keys()
                    if router_name in router_names
                }
            router_data.append(router_results)
            llm_data.append(llm_results)
        # LLM data should be the same for all routers, so just take the first one
        # Pick the llm_data with the most data
        llm_data = sorted(llm_data, key=lambda x: len(x), reverse=True)
        llm_data = llm_data[0]
        router_data_best = []
        for router_results in router_data:
            best_name = []
            cost_performance = []
            for router_variant in router_results.keys():
                best_performance = 0.0
                lowest_cost = 1e10
                router_variant_results = router_results[router_variant]
                # Each variant has a different willingness_to_pay_value
                for willingness_to_pay_value in sorted(router_variant_results):
                    data = router_results[router_variant][willingness_to_pay_value]
                    if data["performance"] > best_performance and not np.isnan(
                        data["performance"]
                    ):
                        best_performance = data["performance"]
                        lowest_cost = data["total_cost"]
                best_name.append(router_variant)
                cost_performance.append((lowest_cost, best_performance))
            # Sort by best performance, then lowest cost
            # Get the indicies, to sort best_name and cost_performance at the same time
            idxes = sorted(
                range(len(cost_performance)),
                key=lambda x: (cost_performance[x][1], -cost_performance[x][0]),
                reverse=True,
            )
            best_name = [best_name[i] for i in idxes]
            cost_performance = [cost_performance[i] for i in idxes]
            if "cascading" in best_name[0]:
                new_best_names = ""
                wanted_error_rate = "0.2"
                for idx, name in enumerate(best_name):
                    # Select best one with the given error rate
                    error_rate = name.split("|")[-2]
                    error_rate = error_rate.split(":")[-1]
                    if error_rate == wanted_error_rate:
                        new_best_names = name
                        break
                best_name = [new_best_names]
            if "svm" not in best_name[0]:
                router_data_best.append({best_name[0]: router_results[best_name[0]]})

        return router_data_best, llm_data

    def plot_performance_vs_cost(
        self,
        eval_name: Optional[str] = None,
        save_file_path: Union[str, bool] = False,
        router_names: list[str] = None,
        show_plot: bool = True,
        plot_ndch: bool = False,
        models_to_route: list[str] = (" ",),
    ):
        """
        Plot the performance vs cost for each eval_name, for all routers by default
        Choose a specific router by specifying router_number, or by passing in the router_names as a list

        :param eval_name: Evaluation name to plot, if None, plots all evals
        :param save_file_path: Path to save out the plot, if desired
        :param show_plot: Whether to show the plot or not
        :param plot_ndch: Whether to plot the non-decreasing convex hull or not
        :param models_to_route: List of model names to route
        :param router_names: List of router names to plot, router name is the base name of the router, e.g. mlp, knn, etc. plus the parameters
            E.g. knn|neighbors:10.0|metric:cosine|fraction:0.1|embedding:all-MiniLM-L12-v2
        """
        router_data_best, llm_data = self._get_performance_vs_cost_plot_data(
            eval_name=eval_name,
            router_names=router_names,
            models_to_route=models_to_route,
        )

        # Plotting routing model data
        self._plot_router_result(
            router_data_best, llm_data, eval_name, save_file_path, show_plot, plot_ndch
        )

    def plot_performance_vs_cost_subplots(
        self,
        eval_names: list[str],
        save_file_path: Union[str, bool] = False,
        router_names: list[str] = None,
        show_plot: bool = True,
        plot_ndch: bool = False,
        models_to_route: list[str] = (" ",),
        ncols: int = 3,
    ):
        nrows = int(np.ceil(len(eval_names) / ncols))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6 * ncols, 4.5 * nrows),
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for ax, eval_name in zip(axes_flat, eval_names):
            router_data_best, llm_data = self._get_performance_vs_cost_plot_data(
                eval_name=eval_name,
                router_names=router_names,
                models_to_route=models_to_route,
            )
            self._plot_router_result(
                router_data_best,
                llm_data,
                eval_name,
                save_file_path=False,
                show_plot=False,
                plot_ndch=plot_ndch,
                ax=ax,
                add_legend=False,
                compact_title=True,
            )

        empty_axes = axes_flat[len(eval_names):]
        for ax in empty_axes:
            ax.axis("off")

        handles, labels = [], []
        for ax in axes_flat[: len(eval_names)]:
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            for handle, label in zip(ax_handles, ax_labels):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)

        fig.suptitle("Cost vs. Performance for Routers and LLMs", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        if len(empty_axes) >= 2:
            left_box = empty_axes[0].get_position()
            right_box = empty_axes[-1].get_position()
            x0 = min(left_box.x0, right_box.x0)
            y0 = min(left_box.y0, right_box.y0)
            x1 = max(left_box.x1, right_box.x1)
            y1 = max(left_box.y1, right_box.y1)
            fig.legend(
                handles,
                labels,
                loc="center",
                bbox_to_anchor=(x0, y0, x1 - x0, y1 - y0),
                bbox_transform=fig.transFigure,
                ncol=2,
            )
        else:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=4,
            )
            fig.tight_layout(rect=(0, 0.12, 1, 0.97))

        if save_file_path:
            assert isinstance(
                save_file_path, str
            ), "save_file_path must be a string if not False"
            graph_name = (
                f"all_routers_{generate_datetime_str()}_combined_{plot_ndch}"
                "_performance_vs_cost.png"
            )
            fig.savefig(os.path.join(save_file_path, graph_name), bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def _plot_router_result(
        self,
        router_data,
        llm_data,
        eval_name,
        save_file_path: Union[str, bool] = False,
        show_plot=True,
        plot_ndch=False,
        ax=None,
        add_legend=True,
        compact_title=False,
    ):
        owns_figure = ax is None
        if owns_figure:
            _, ax = plt.subplots(figsize=(12, 6))
        # Create list of 14 colors to be used for matplotlib plotting
        NUM_COLORS = sum([len(router_results) for router_results in router_data]) + len(
            llm_data.items()
        )
        color_map = plt.get_cmap("tab20c")
        colors = [color_map(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
        for router_idx, router_results in enumerate(router_data):
            # All the variants of the router exist here
            for router_variant in router_results.keys():
                router_variant_results = router_results[router_variant]
                x_coords, y_coords = [], []
                # Each variant has a different willingness_to_pay_value
                for willingness_to_pay_value in sorted(router_variant_results):
                    data = router_results[router_variant][willingness_to_pay_value]
                    x_coords.append(data["total_cost"])
                    y_coords.append(data["performance"])
                # Remove duplicates
                x_coords, y_coords = zip(*sorted(set(zip(x_coords, y_coords))))
                router_name_display = router_variant.split("|")[0]
                if "mlp" in router_name_display or "knn" in router_name_display:
                    router_name_display = router_name_display.upper()
                else:
                    router_name_display = "Cascading"

                if plot_ndch:
                    try:
                        point_array = np.stack(
                            (np.array(x_coords), np.array(y_coords)), axis=-1
                        )
                        ndch_points = get_non_descreasing_convex_hull_of(point_array)
                        ax.plot(
                            ndch_points[:, 0],
                            ndch_points[:, 1],
                            label=f"{router_name_display} NDCH",
                            linestyle="solid",
                            color=colors[router_idx],
                        )
                    except Exception as e:
                        print("Could not plot NDCH for", router_name_display)
                        ax.plot(
                            x_coords,
                            y_coords,
                            label=f"{router_name_display} NDCH",
                            linestyle="solid",
                            color=colors[router_idx],
                        )
                if "cascading router" in router_name_display:
                    error_rate = router_variant.split("|")[-2].split(":")[-1]
                    full_display_name = f"cascading router error: {error_rate}"
                else:
                    full_display_name = f"{router_name_display} router"

                ax.plot(
                    x_coords,
                    y_coords,
                    label=full_display_name,
                    linestyle="dotted",
                    marker="x",
                    color=colors[router_idx],
                )
        # Adding LLM data as scatter points
        llm_xs = []
        llm_ys = []
        llm_color_offset = sum(len(router_results) for router_results in router_data)
        for idx, (model_name, values) in enumerate(llm_data.items()):
            if model_name in MODELS_TO_ROUTE_NAMES:
                label = MODELS_TO_ROUTE_NAMES[model_name]
            else:
                label = model_name
            if model_name != "oracle":
                llm_xs.append(values[0])
                llm_ys.append(values[1])
            if model_name == "oracle":
                ax.scatter(
                    values[0],
                    values[1],
                    label=label,
                    color=colors[llm_color_offset + idx],
                    marker="*",
                    s=40,
                )
            else:
                ax.scatter(
                    values[0],
                    values[1],
                    label=label,
                    color=colors[llm_color_offset + idx],
                )

        # Add zero router
        if llm_xs:
            point_array = np.stack((np.array(llm_xs), np.array(llm_ys)), axis=-1)
            ndch_points = get_non_descreasing_convex_hull_of(point_array)
            ax.plot(
                ndch_points[:, 0],
                ndch_points[:, 1],
                label="Zero router",
                linestyle="solid",
                color="grey",
            )

        # Final plot formatting
        ax.set_xlabel("Total Cost ($)")
        ax.set_ylabel("Performance")
        if eval_name is not None:
            if compact_title:
                ax.set_title(eval_name)
            else:
                ax.set_title(
                    "Cost vs. Performance for Routers and LLMs\neval: {}".format(
                        eval_name
                    )
                )
        else:
            ax.set_title("Cost vs. Performance for Routers and LLMs\neval: all")
        if add_legend:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(True)
        ax.set_ylim(0, 1)
        if save_file_path:
            assert isinstance(
                save_file_path, str
            ), "save_file_path must be a string if not False"
            eval_name_in_path = eval_name if eval_name is not None else "all"
            graph_name = f"all_routers_{generate_datetime_str()}_{eval_name_in_path}_{plot_ndch}_performance_vs_cost.png"
            ax.figure.tight_layout()
            ax.figure.savefig(os.path.join(save_file_path, graph_name))
        if show_plot:
            plt.show()
        elif owns_figure:
            plt.close(ax.figure)
