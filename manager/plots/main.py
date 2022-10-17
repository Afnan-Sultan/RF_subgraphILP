import os
from math import ceil

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from manager.data.post_process_results import postprocessing

matplotlib.use("TkAgg")

col_map = {
    "corr_num": "lemonchiffon",
    "corr_num_bias": "khaki",
    "corr_num_sauron": "darkkhaki",
    "corr_num_bias_sauron": "olive",
    "corr_thresh": "paleturquoise",
    "corr_thresh_bias": "mediumturquoise",
    "corr_thresh_sauron": "lightseagreen",
    "corr_thresh_bias_sauron": "teal",
    "subILP": "thistle",
    "subILP_bias": "plum",
    "subILP_sauron": "violet",
    "subILP_bias_sauron": "purple",
    "targeted_subILP": "lightsteelblue",
    "targeted_subILP_bias": "cornflowerblue",
    "targeted_subILP_sauron": "royalblue",
    "targeted_subILP_bias_sauron": "navy",
    "rf": "peachpuff",
    "rf_sauron": "sandybrown",
    "random": "lightgrey",
    "random_bias": "grey",
}


def box_plot(
    dfs,
    title,
    subtitles,
    output_dir,
    regression,
    num_cols,
    figsize=(10, 10),
    showfliers=False,
    fontsize=12,
    rotation=30,
    ha="right",
    hspace=0.5,
):
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title)
    ascending = True if regression else False
    legends_box = None
    col_order = None
    legends_label = None
    y_max = 0
    for idx, df in enumerate(dfs):
        df = df.round(2)
        if col_order is None:
            to_sort = pd.DataFrame(
                {"mean": df.mean(), "median": df.median()}, index=df.columns
            ).sort_values(by=["mean", "median"], ascending=[ascending, ascending])
            col_order = to_sort.index.to_list()
            legends_label = col_order.copy()
            legends_label.extend(["mean", "median"])
        df = df.loc[:, col_order]

        ax = fig.add_subplot(ceil(len(dfs) / num_cols), num_cols, idx + 1)
        bplot = ax.boxplot(
            df.values, showfliers=showfliers, showmeans=True, patch_artist=True
        )

        colors = [col_map[model] for model in col_order]
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
        legends_box = [bplot["boxes"][idx] for idx in range(len(bplot["boxes"]))]
        legends_box.extend([bplot["means"][0], bplot["medians"][0]])

        if ax.get_ylim()[1] > y_max:
            y_max = ax.get_ylim()[1]
            for prev_idx in range(idx + 1):
                prev_ax = fig.get_axes()[prev_idx]
                prev_ax.set_ylim(top=y_max)
        ax.set_ylim(top=y_max)
        ax.set_xticklabels(df.columns, fontsize=fontsize)
        ax.set_title(subtitles[idx])
        ax.axes.xaxis.set_visible(False)
        plt.xticks(rotation=rotation, ha=ha)
    bottom_loc = fig.subplotpars.bottom
    plt.subplots_adjust(bottom=bottom_loc + (0.5 * bottom_loc))
    bottom_loc = fig.subplotpars.bottom
    right_loc = fig.subplotpars.right
    plt.legend(
        legends_box,
        legends_label,
        fontsize=fontsize,
        bbox_to_anchor=(right_loc, bottom_loc - 0.01),
        ncols=5,
        bbox_transform=fig.transFigure,
    )
    # plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title}.png"))


if __name__ == "__main__":
    results_path = "../../results_larger"
    output_path = "../../figures_v3"

    regression = True
    weighted = True
    simple_weight = True
    targeted = True
    n_features = 20

    if weighted and simple_weight:
        dir0 = "weighted"
    elif weighted and not simple_weight:
        dir0 = "weighted_linear"
    else:
        dir0 = "not_weighted"
    if regression:
        dir1 = "regression"
    else:
        dir1 = "classification"
    if targeted:
        dir2 = "targeted"
    else:
        dir2 = "not_targeted"
    output_dir_ = os.path.join(output_path, dir0, dir1, dir2)
    os.makedirs(output_dir_, exist_ok=True)

    ml_metrics = {
        "regression": {
            "overall": "MSE scores",
            "sensitivity": "MSE scores - sensitive",
            "specificity": "MSE scores - resistant",
        },
        "classification": {
            "sensitivity": "sensitivity",
            "specificity": "specificity",
            "f1": "F1-score",
            "youden_j": "Youden's J",
            "mcc": "Matthews Correlation Coefficient",
        },
    }
    if regression:
        metrics = ml_metrics["regression"]
        thresh = 0.1
    else:
        thresh = 0.01
        metrics = ml_metrics["classification"]

    # results postprocessing
    condition = f"{dir1}_{dir0}"
    (
        parameters_grid,
        final_models_acc,
        final_models_parameters,
        final_models_best_features,
        final_models_num_features,
        metric_best_models,
        parameters_grid_acc,
        parameters_grid_acc_per_drug,
        cross_validation_splits_acc,
        runtimes,
    ) = postprocessing(
        results_path, condition, targeted, regression, list(metrics.keys()), thresh
    )

    for metric in metrics:
        scores = [
            final_models_acc[metric]["test_score"],
            final_models_acc[metric]["train_score"],
        ]
        subtitles_ = ["test scores", "train_scores"]
        box_plot(
            dfs=scores,
            title=f"{metrics[metric]}",
            subtitles=subtitles_,
            output_dir=output_dir_,
            regression=regression,
            num_cols=2,
            figsize=(15, 12),
            rotation=30,
        )
