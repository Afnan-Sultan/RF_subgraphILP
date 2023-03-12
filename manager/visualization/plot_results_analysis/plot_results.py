import os
from math import ceil

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cbook import boxplot_stats
from tqdm import tqdm

matplotlib.use("Agg")


def sort_models(df, ascending, bar, legends_label):
    # rearrange models from best to worst
    if bar:
        if isinstance(df, pd.DataFrame):
            col_order = df.sort_values(
                by=df.columns[0], axis=0, ascending=ascending
            ).index.to_list()
        else:
            col_order = df.sort_values(ascending=ascending).index.to_list()
    else:
        to_sort = pd.DataFrame(
            {"mean": df.mean(), "median": df.median()}, index=df.columns
        ).sort_values(by=["mean", "median"], ascending=[ascending, ascending])
        col_order = to_sort.index.to_list()
    if legends_label is None:
        legends_label = col_order.copy()
        if not bar:
            legends_label.extend(["mean", "median"])
    return col_order, legends_label


def plot_columns(
    scores,
    ascending,
    col_order,
    bar,
    row_idx,
    axes,
    subtitles,
    legends_label,
    legends_box,
    col_map,
    showfliers,
    row_title_xpos,
    row_title_ypos,
    fontsize,
    sharey=False,
    ylim=None,
    ylabel=None,
    share_ax_text=False,
    share_ax_title=False,
    to_rename=None,
    metric=None,
    ax_text_rot=0,
):
    for col_idx, df in enumerate(scores):
        # round to get rid of miniscule differences
        df = df.round(2).dropna()

        # rearrange models from best to worse
        if col_order is None:
            col_order, legends_label = sort_models(df, ascending, bar, legends_label)

        if bar:
            df = df.loc[col_order]
        else:
            df = df.loc[:, col_order]

        # specify a color for each model from a predefined dictionary map
        colors = [col_map[model] for model in col_order]

        if axes.ndim > 1:
            ax = axes[row_idx][col_idx]
        else:
            ax = axes[row_idx]
        ax.axes.xaxis.set_visible(False)

        if bar:
            bplot = ax.bar(df.index.to_list(), df.values)
            models_plots = bplot
        else:
            bplot = ax.boxplot(
                df.values,
                showfliers=showfliers,
                showmeans=True,
                patch_artist=True,
            )
            models_plots = bplot["boxes"]
        for patch, color in zip(models_plots, colors):
            patch.set_facecolor(color)

        # store each boxplot object to be for legend retrieval
        if legends_box is None:
            legends_box = [models_plots[idx] for idx in range(len(models_plots))]
            if not bar:
                legends_box.extend([bplot["means"][0], bplot["medians"][0]])

        # add x and y labels for the starting row and columns only of the figure
        if share_ax_title:
            if row_idx == 0:
                ax.set_title(subtitles[col_idx], fontsize=fontsize)
        else:
            ax.set_title(subtitles[col_idx], fontsize=fontsize)

        if ylim is not None and sharey == "row":
            ax.set_ylim(ylim)
            y = (ylim[1] - ylim[0]) / 2
            y += row_title_ypos * (y - ylim[0])
        else:
            ylim = ax.get_ylim()
            y = ylim[1] - (row_title_ypos * (ylim[1] - ylim[0]))
        xlim = ax.get_xlim()
        if share_ax_text and to_rename is not None:
            if col_idx == 0:
                ax.text(
                    x=xlim[0] - (row_title_xpos * xlim[1]),
                    y=y,
                    s=to_rename[metric],
                    rotation=ax_text_rot,
                    fontsize=fontsize,
                )
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
    return legends_box, legends_label, col_order


def plot_averaged(
    acc_dict,
    by_count_dict,
    title,
    fig_name,
    col_map,
    output_dir,
    to_rename,
    metrics,
    showfliers=False,
    output_cv=True,
    regression=None,
    figsize=(10, 10),
    fontsize=16,
    hspace=0.1,
    wspace=0.1,
    legend_ncol=1,
    legened_pos_right=0.0,
    change_bottom_by=0.0,
    change_right_by=0.0,
    row_title_xpos=0.0,
    row_title_ypos=0.0,
    ax_text_rot=0,
):
    fig_ncols = 2
    fig_nrows = len(metrics)
    fig, axes = plt.subplots(fig_nrows, fig_ncols, figsize=figsize, sharey=False)
    plt.suptitle(title, fontsize=fontsize + 4)
    # adjust figure's layout. shrinkage of borders is for the legend sake.
    bottom_loc = fig.subplotpars.bottom
    right_loc = fig.subplotpars.right
    plt.subplots_adjust(
        right=right_loc - (change_right_by * right_loc),
        bottom=bottom_loc - (change_bottom_by * bottom_loc),
        hspace=hspace,
        wspace=wspace,
    )
    # bottom_loc = fig.subplotpars.bottom
    # left_loc = fig.subplotpars.left
    right_loc = fig.subplotpars.right
    top_loc = fig.subplotpars.top

    ascending = True if regression else False
    col_order = None
    legends_box = None
    legends_label = None
    to_csv = pd.DataFrame()
    for row_idx, metric in enumerate(metrics):
        acc_df = acc_dict[metric].abs().round(2)

        if output_cv:
            scores_summary = scores_percentiles(
                acc_df, acc_df.columns.to_list(), metric
            )
            to_csv = pd.concat([to_csv, scores_summary])

        if col_order is None:
            col_order, legends_label = sort_models(
                acc_df, ascending, False, legends_label
            )
        # specify a color for each model from a predefined dictionary map
        colors = [col_map[model] for model in col_order]

        acc_df = acc_df.loc[:, col_order]
        ax = axes[row_idx, 0]
        ax.axes.xaxis.set_visible(False)
        if row_idx == 0:
            ax.set_title("Average test scores", fontsize=fontsize)
        if regression:
            ax.set_ylabel("MSE scores")
        else:
            ax.set_ylim(-0.05, 1.05)
        bplot = ax.boxplot(
            acc_df.values,
            showfliers=showfliers,
            showmeans=True,
            patch_artist=True,
        )
        models_plots = bplot["boxes"]
        for patch, color in zip(models_plots, colors):
            patch.set_facecolor(color)

        ylim = ax.get_ylim()
        y = ylim[1] - (row_title_ypos * ylim[1])
        xlim = ax.get_xlim()
        x = xlim[0] - (row_title_xpos * xlim[1])
        ax.text(
            x=x,
            y=y,
            s=to_rename[metric],
            rotation=ax_text_rot,
            fontsize=fontsize,
            horizontalalignment="right",
            verticalalignment="center",
            # transform=ax.transAxes,
        )

        # store each boxplot object to be used for legend retrieval
        if legends_box is None:
            legends_box = [models_plots[idx] for idx in range(len(models_plots))]
            legends_box.extend([bplot["means"][0], bplot["medians"][0]])

        by_count_df = by_count_dict[metric].loc[col_order]["test_score"]
        ax = axes[row_idx, 1]
        ax.axes.xaxis.set_visible(False)
        if row_idx == 0:
            ax.set_title("One Standard Error", fontsize=fontsize)
        ax.set_ylabel("Number of drugs")
        bplot = ax.bar(by_count_df.index.to_list(), by_count_df.values)
        models_plots = bplot
        for patch, color in zip(models_plots, colors):
            patch.set_facecolor(color)

    if output_cv:
        to_csv.round(2).to_csv(os.path.join(output_dir, f"{fig_name}.csv"))

    # add the legend retrieved from the first subplot (the names and colors are preserved for the remaining subplots)
    right_loc = fig.subplotpars.right
    top_loc = fig.subplotpars.top
    plt.legend(
        legends_box,
        legends_label,
        fontsize=fontsize,
        bbox_to_anchor=(right_loc + legened_pos_right, top_loc),
        ncols=legend_ncol,
        bbox_transform=fig.transFigure,
    )
    plt.savefig(os.path.join(output_dir, f"{fig_name}.png"))  # , format="svg"
    plt.close()


def plot_accuracies(
    models_acc,
    title,
    fig_name,
    col_map,
    output_dir,
    fig_ncols,
    to_rename=None,
    metrics=None,
    output_cv=False,
    subtitles=None,
    ascending=None,
    final=False,
    in_minutes=False,
    regression=None,
    single_level_df=False,
    multi_level_df=False,
    averaged_results=False,
    sharey=False,
    ylabel=None,
    bar=False,
    figsize=(10, 10),
    showfliers=False,
    fontsize=16,
    hspace=0.1,
    wspace=0.1,
    legend_ncol=1,
    legened_pos_right=0.0,
    change_bottom_by=0.0,
    change_right_by=0.0,
    row_title_xpos=0.0,
    row_title_ypos=0.0,
    ax_text_rot=0,
):
    os.makedirs(output_dir, exist_ok=True)

    if averaged_results:
        # for accuracy per drug in the averaged test scores mode
        fig_ncols = 1
        figsize = (int(figsize[0] * 0.85), figsize[1])

    # initialize a figure
    if single_level_df:
        # should be invoked when plotting runtimes
        fig_nrows = ceil(len(metrics) / fig_ncols)
        # if not final:
        #     fig_nrows = len(metrics)
    elif multi_level_df:
        fig_nrows = 1
    else:
        fig_nrows = len(metrics)
    fig, axes = plt.subplots(fig_nrows, fig_ncols, figsize=figsize, sharey=sharey)

    # adjust figure's layout. shrinkage of borders is for the legend sake.
    bottom_loc = fig.subplotpars.bottom
    right_loc = fig.subplotpars.right
    plt.subplots_adjust(
        right=right_loc - (change_right_by * right_loc),
        bottom=bottom_loc - (change_bottom_by * bottom_loc),
        hspace=hspace,
        wspace=wspace,
    )
    # bottom_loc = fig.subplotpars.bottom
    # left_loc = fig.subplotpars.left
    right_loc = fig.subplotpars.right
    top_loc = fig.subplotpars.top

    # variables to be used for the overall figure
    plt.suptitle(title, fontsize=fontsize + 4)
    col_order = None
    legends_box = None
    legends_label = None

    # sorting criteria for the graph. lower is better for regression while higher is better for classification
    if ascending is None:
        if regression is not None:
            ascending = True if regression else False
        else:
            ascending = False

    if single_level_df:
        all_scores = [models_acc[col] for col in metrics]
        if in_minutes:
            all_scores = [scores / 60 for scores in all_scores]
        if to_rename is not None:
            all_subtitles = [to_rename[col] for col in metrics]
        else:
            all_subtitles = metrics
        for row_idx in range(fig_nrows):
            scores = all_scores[:fig_ncols]
            all_scores = all_scores[fig_ncols:]
            subtitles = all_subtitles[:fig_ncols]
            all_subtitles = all_subtitles[fig_ncols:]
            legends_box, legends_label, col_order = plot_columns(
                scores,
                ascending,
                col_order,
                bar,
                row_idx,
                axes,
                subtitles,
                legends_label,
                legends_box,
                col_map,
                showfliers,
                row_title_xpos,
                row_title_ypos,
                fontsize,
                sharey=sharey,
            )
    elif multi_level_df:
        row_idx = 0
        col_0 = models_acc.columns.levels[0]
        scores = [models_acc[col] for col in col_0]
        if subtitles is None:
            subtitles = col_0
        ylim = get_ylim(
            sharey, bar, scores, regression, models_acc, subtitles, metric=None
        )
        legends_box, legends_label, col_order = plot_columns(
            scores,
            ascending,
            col_order,
            bar,
            row_idx,
            axes,
            subtitles,
            legends_label,
            legends_box,
            col_map,
            showfliers,
            row_title_xpos,
            row_title_ypos,
            fontsize,
            sharey=sharey,
            ylim=ylim,
            ylabel=ylabel,
            share_ax_text=False,
            share_ax_title=True,
            to_rename=to_rename,
            ax_text_rot=ax_text_rot,
        )
    else:
        to_csv = pd.DataFrame()
        for row_idx, metric in enumerate(metrics):
            if final:
                scores = [models_acc[metric]]
                if subtitles is None:
                    subtitles = ["Test_scores"]
            elif averaged_results:
                scores = [models_acc[metric]["test_score"]]
                subtitles = ["Average Test Scores"]
            else:
                if models_acc[metric].columns.nlevels > 1:
                    cols_0 = models_acc[metric].columns.levels[0]
                else:
                    cols_0 = models_acc[metric].columns
                scores = [models_acc[metric][col].abs() for col in cols_0]
                if subtitles is None:
                    subtitles = cols_0

            if output_cv:
                scores_summary = scores_percentiles(models_acc, subtitles, metric)
                to_csv = pd.concat([to_csv, scores_summary])

            ylim = get_ylim(
                sharey, bar, scores, regression, models_acc, subtitles, metric=metric
            )
            legends_box, legends_label, col_order = plot_columns(
                scores,
                ascending,
                col_order,
                bar,
                row_idx,
                axes,
                subtitles,
                legends_label,
                legends_box,
                col_map,
                showfliers,
                row_title_xpos,
                row_title_ypos,
                fontsize,
                sharey=sharey,
                ylim=ylim,
                ylabel=ylabel,
                share_ax_text=True,
                share_ax_title=True,
                to_rename=to_rename,
                metric=metric,
                ax_text_rot=ax_text_rot,
            )
        if output_cv:
            to_csv.round(2).to_csv(os.path.join(output_dir, f"{fig_name}.csv"))

    # add the legend retrieved from the first subplot (the names and colors are preserved for the remaining subplots)
    plt.legend(
        legends_box,
        legends_label,
        fontsize=fontsize,
        bbox_to_anchor=(right_loc + legened_pos_right, top_loc),
        ncols=legend_ncol,
        bbox_transform=fig.transFigure,
    )
    plt.savefig(os.path.join(output_dir, f"{fig_name}.png"))  # , format="svg"
    plt.close()


def calc_percentile(model, df_model, metric_summary):
    summary = pd.DataFrame(boxplot_stats(df_model.values))
    summary = summary[["whislo", "med", "mean", "whishi", "iqr"]]
    summary.rename(
        {
            "whislo": "$Q_0$",
            "med": "Med",
            "mean": "Mean",
            "whishi": "$Q_4$",
            "iqr": "$IQR$",
        },
        axis=1,
        inplace=True,
    )
    metric_summary[model] = summary.loc[0]


def scores_percentiles(models_acc, subtitles, metric=None):
    if isinstance(models_acc, pd.DataFrame):
        df = models_acc
        cols = df.columns
    else:
        df = models_acc[metric]
        cols = df.columns.levels[0]
    scores_summary = pd.DataFrame()
    assert len(cols) == len(subtitles)
    for col, name in {cols[i]: subtitles[i] for i in range(len(subtitles))}.items():
        metric_summary = pd.DataFrame()
        metric_acc = df[col]
        if isinstance(metric_acc, pd.Series):
            calc_percentile(col, metric_acc, metric_summary)
            scores_summary = pd.concat([scores_summary, metric_summary], axis=1)
        else:
            for model in metric_acc:
                df_model = metric_acc[model]
                calc_percentile(model, df_model, metric_summary)
            metric_summary = pd.concat(
                [metric_summary.transpose()], axis=1, keys=[name]
            )
            scores_summary = pd.concat([scores_summary, metric_summary], axis=1)
    if metric is not None:
        scores_summary = pd.concat([scores_summary], keys=[metric])
    return scores_summary


def get_ylim(sharey, bar, scores, regression, models_acc, subtitles, metric=None):
    ylim = None
    if sharey == "row":
        if bar:
            ylim = (
                0,
                max([df.max() for df in scores]) + 1,
            )
        else:
            if regression:
                temp = scores_percentiles(models_acc, subtitles, metric).droplevel(
                    0, axis=1
                )
                ymin = min(temp["$Q_0$"].min()) - 0.2
                ymax = max(temp["$Q_4$"].max()) + 0.5
            else:
                ymin = -0.05
                ymax = 1.05
            ylim = (ymin, ymax)
    return ylim


def plot_splits(
    drugs,
    fig_title,
    fig_name,
    fig_ncols,
    output_dir,
    to_rename=None,
    xlabel=None,
    hspace=0.2,
    wspace=0.2,
    figsize=(18, 5),
    fontsize=12,
    legened_pos=(0, 0),
    legend_ncol=1,
    sharey=False,
):
    os.makedirs(output_dir, exist_ok=True)
    for drug, models in tqdm(drugs.items(), desc="plotting splits per drugs"):
        to_be_used = {
            model: res
            for model, res in models.items()
            if "random" not in model and "rf" not in model
        }
        possible_nrows = ceil(len(to_be_used.keys()) / fig_ncols)
        fig_nrows = ceil(len(to_be_used.keys()) / fig_ncols)
        fig, axes = plt.subplots(fig_nrows, fig_ncols, sharey=sharey)
        fig_idx = 0
        legend_box = None
        legend_labels = None
        sorted_models = sorted(list(to_be_used.keys()))
        for model in sorted_models:
            df = to_be_used[model]
            renamed_index = {idx: idx.replace("split", "fold") for idx in df.index}
            df = df.rename(renamed_index)
            if to_rename is not None:
                df = df.rename(to_rename, axis=1)
            ax = plt.subplot(fig_nrows, fig_ncols, fig_idx + 1)
            df.plot(ax=ax, title=model, figsize=figsize)
            if isinstance(df, pd.DataFrame) and df.ndim > 1:
                if legend_box is None:
                    legend_box, legend_labels = plt.gca().get_legend_handles_labels()
                ax.get_legend().remove()

            if xlabel is not None:
                ax.set_xlabel(xlabel)

            fig_idx += 1

        plt.suptitle(f"{drug} - {fig_title}")
        plt.subplots_adjust(wspace=wspace, hspace=hspace, top=0.9)

        if legend_box is not None:
            plt.legend(
                legend_box,
                legend_labels,
                fontsize=fontsize,
                bbox_to_anchor=(legened_pos[0], legened_pos[1]),
                ncols=legend_ncol,
                # bbox_transform=plt.transFigure,
            )
        plt.savefig(
            os.path.join(output_dir, f"{drug}_{fig_name}.png"),
            bbox_inches="tight",
        )
        plt.close()


def plot_trees_dist(
    drugs,
    fig_title,
    fig_name,
    output_dir,
    col_map,
    figsize=(18, 5),
):
    os.makedirs(output_dir, exist_ok=True)
    for drug, models in tqdm(drugs.items(), desc="plotting trees features per drug"):
        sorted_models = sorted(list(models.keys()))
        # lines = {}
        for model in sorted_models:
            df = models[model]  # type: pd.Series
            if model == "subgraphilp":
                model = "subILP"
            if "bias" in fig_title:
                model = f"{model}_bias"
            df.rename(model, inplace=True)
            df.plot(
                figsize=figsize,
                color=col_map[model],
                legend=model,
                ylabel="number of trees with feature",
                xlabel="features alias",
            )
        plt.title(f"{drug} - {fig_title}")
        plt.savefig(
            os.path.join(output_dir, f"{drug}_{fig_name}.png"),
            bbox_inches="tight",
        )
        plt.close()


def plot_common_features_mat(
    df,
    drug,
    output_dir,
    name,
    n_features=None,
    figsize=(7.7, 6),
    selected=False,
    title_pos=0.85,
    limits=[0, 100],
):
    os.makedirs(output_dir, exist_ok=True)
    alpha = df.index.to_list()
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=title_pos)
    ax_title = f"{name} - {drug}"
    if selected:
        title = "Percentage of common selected features"
        fig_name = f"{drug}_selected_features_mat.png"
    else:
        if n_features is None:
            n_features = 100
        title = f"Number of common best features (best {n_features})"
        fig_name = f"{drug}_important_features_mat.png"
    plt.suptitle(title)
    ax = fig.add_subplot(111)
    ax.set_title(ax_title, fontsize=10)
    cax = ax.matshow(df, interpolation="nearest", vmin=limits[0], vmax=limits[1])
    fig.colorbar(cax)
    ax.set_xticklabels([""] + alpha)
    ax.set_yticklabels([""] + alpha)
    plt.savefig(os.path.join(output_dir, fig_name))
    plt.close()


def plot_single_boxplot(df, col_map, col_order, output_dir, fig_name, figsize):
    fig = plt.figure(figsize=figsize)
    plt.suptitle("Number of Unique Features", fontsize=20)
    right_loc = fig.subplotpars.right
    top_loc = fig.subplotpars.top
    ax = fig.add_subplot(111)
    bplot = ax.boxplot(
        df.values,
        showfliers=True,
        showmeans=True,
        patch_artist=True,
    )
    models_plots = bplot["boxes"]
    colors = [col_map[model] for model in col_order]
    for patch, color in zip(models_plots, colors):
        patch.set_facecolor(color)

    xtickNames = plt.setp(ax, xticklabels=col_order)
    plt.setp(xtickNames, rotation=35, fontsize=12)
    legends_box = [models_plots[idx] for idx in range(len(models_plots))]
    legends_box.extend([bplot["means"][0], bplot["medians"][0]])
    plt.savefig(os.path.join(output_dir, f"{fig_name}.png"))  # , format="svg"
    plt.close()


def plot_params_acc(
    param_acc,
    params_df,
    output_dir,
    renamed_metrics,
    figsize=(20, 15),
    fig_ncols=2,
    transparent=False,
    row_title_ypos=0.2,
    row_title_xpos=0.13,
):
    os.makedirs(output_dir, exist_ok=True)
    fig_nrows = len(param_acc)
    fig, axes = plt.subplots(fig_nrows, fig_ncols, sharey="row", figsize=figsize)
    plt.subplots_adjust(top=0.93)
    plt.suptitle(f"Parameters Tuning", fontsize=20)
    for row_idx, metric in enumerate(list(param_acc.keys())):
        df = param_acc[metric].sort_index().groupby(level=0).mean()
        df.index = df.index.astype(int)

        max_features = params_df.groupby("max_features").groups
        for col_idx, model in enumerate(df.columns):
            ax = axes[row_idx, col_idx]
            new_df = pd.DataFrame()
            for max_feat, indices in max_features.items():
                new_df[max_feat] = df[model].iloc[list(indices)].to_list()
            new_df.index = params_df["min_samples_leaf"].iloc[new_df.index]
            new_df.plot(ax=ax)
            ax.set_xlabel("minimum samples per leaf")

            ylim = ax.get_ylim()
            y = ylim[1] - (row_title_ypos * (ylim[1] - ylim[0]))
            xlim = ax.get_xlim()
            if col_idx == 0:
                ax.text(
                    x=xlim[0] - (row_title_xpos * xlim[1]),
                    y=y,
                    s=renamed_metrics[metric],
                    rotation=0,
                    fontsize=20,
                )
            if row_idx == 0:
                ax.set_title(model, fontsize=15)

    plt.savefig(
        os.path.join(output_dir, f"parameter_tuning.png"),
        bbox_inches="tight",
        transparent=transparent,
    )
    plt.close()


def plot_one_se(
    one_se_data,
    renamed_metrics,
    to_arrange,
    output_dir,
    renamed_analysis,
    analysis_type=None,
):
    for element in one_se_data:
        if os.path.isfile(element):
            analysis_type = os.path.basename(element).split(".")[0]
            analysis_type = "_".join(analysis_type.split("_")[:-3])
            data = pd.read_csv(element, index_col=[0, 1, 2], skipinitialspace=True)
            output_img = os.path.join(output_dir, f"{analysis_type}_filter_summary.png")
        else:
            assert analysis_type is not None
            output_img = os.path.join(output_dir, f"{analysis_type}_filter_summary.png")
            data = element

        existing_metrics = [m for m in to_arrange if m in data.index.levels[0]]
        data = data.loc[existing_metrics]

        fig, axes = plt.subplots(1, len(data.index.levels[0]))
        plt.suptitle(
            f"Different One Standard Error Filters - {renamed_analysis[analysis_type]}"
        )
        for idx, metric in enumerate(existing_metrics):
            metric_df = data.loc[metric]
            drug_counts = metric_df.groupby(level=[1], axis=0).count()
            acc = drug_counts["acc"]
            num_features = acc - drug_counts["num_features"]
            runtimes = num_features - drug_counts["runtime"]
            filters_df = pd.concat(
                [
                    acc.rename("accuracy"),
                    num_features.rename("+ num_features"),
                    runtimes.rename("+ runtime"),
                ],
                axis=1,
            )

            ax = axes[idx]
            ax.set_ylabel("number of drugs")
            filters_df.plot(
                kind="bar",
                ax=ax,
                title=f"{renamed_metrics[metric]} - One Standard Error",
                figsize=(20, 5),
                rot=75,
                align="center",
            )
        plt.savefig(
            output_img,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    figures_dir = "/home/afnan/Afnan/Thesis/RF_subgraphILP/figures_v4/weighted/regression/not_targeted"
    files = [
        os.path.join(figures_dir, file)
        for file in os.listdir(figures_dir)
        if "filter_summary.csv" in file
    ]
    renamed_metrics_ = {
        "overall": "$MSE$",
        "sensitivity": "$MSE_{sens}$",
        "specificity": "$MSE_{res}$",
    }
    renamed_analysis_ = {
        "rf_vs_subilp": "Regression Performance",
        "data_vs_prior": "Regression Methods Comparison",
        "without_bias": "Regression Bias Ablation",
        "without_synergy": "Regression Synergy Ablation",
        "with_sauron": "Regression With SAURON",
    }
    arranged_metrics = ["sensitivity", "specificity", "overall"]
    plot_one_se(
        files, renamed_metrics_, arranged_metrics, figures_dir, renamed_analysis_
    )
