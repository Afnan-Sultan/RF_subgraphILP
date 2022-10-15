import os
from math import ceil

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from manager.data.post_process_results import postprocessing

matplotlib.use("TkAgg")


def box_plot(
    dfs,
    title,
    subtitles,
    output_dir,
    regression,
    figsize=(10, 10),
    showfliers=False,
    fontsize=12,
    rotation=30,
    ha="right",
):
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=hspace)
    plt.suptitle(title)
    for idx, df in enumerate(dfs):
        df = df.round(2)
        ascending = True if regression else False
        to_sort = pd.DataFrame(
            {"median": df.median(), "q3": df.quantile(0.75)}, index=df.columns
        ).sort_values(by=["median", "q3"], ascending=[ascending, ascending])
        df = df.loc[:, to_sort.index]
        ax = fig.add_subplot(ceil(len(dfs) / 2), 2, idx + 1)
        ax.boxplot(df.values, showfliers=showfliers)
        ax.set_xticklabels(df.columns, fontsize=fontsize)
        ax.set_title(subtitles[idx])
        plt.xticks(rotation=rotation, ha=ha)
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}_boxplot.png"))


if __name__ == "__main__":
    results_path = "../../results_larger"
    output_path = "../../figures_v3"

    regression = False
    weighted = True
    simple_weight = True
    targeted = True
    n_features = 20
    hspace = 1.2

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
    else:
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
        results_path,
        condition,
        targeted,
        regression,
        list(metrics.keys()),
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
            figsize=(15, 10),
        )
