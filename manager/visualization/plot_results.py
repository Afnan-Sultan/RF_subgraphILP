import os
from math import ceil

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cbook import boxplot_stats

matplotlib.use("Agg")


def sort_models(df, ascending, bar, legends_label):
    # rearrange models from best to worse
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
    ylim=None,
    ylabel=None,
    share_ax_text=False,
    share_ax_title=False,
    to_rename=None,
    metric=None,
    ax_text_rot=90,
):
    for col_idx, df in enumerate(scores):
        # round to get rid of miniscule differences
        df = df.round(2)

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

        if ylim is not None:
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
    in_minutes=True,
    regression=None,
    single_level_df=False,
    multi_level_df=False,
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
    ax_text_rot=90,
):
    os.makedirs(output_dir, exist_ok=True)

    # initialize a figure
    if single_level_df:
        if not final:
            fig_nrows = len(metrics)
        # should be invoked when plotting runtimes
        fig_nrows = ceil(len(metrics) / fig_ncols)
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
            ylim=ylim,
            ylabel=ylabel,
            share_ax_text=True,
            share_ax_title=True,
            to_rename=to_rename,
            ax_text_rot=ax_text_rot,
        )
    else:
        to_csv = pd.DataFrame()
        for row_idx, metric in enumerate(metrics):
            if output_cv:
                scores_summary = scores_percentiles(models_acc, metric)
                to_csv = pd.concat([to_csv, scores_summary])

            if final:
                scores = [models_acc[metric]]
                if subtitles is None:
                    subtitles = ["Test_scores"]
            else:
                assert models_acc[metric].columns.nlevels > 1
                cols_0 = models_acc[metric].columns.levels[0]
                scores = [models_acc[metric][col].abs() for col in cols_0]
                if subtitles is None:
                    subtitles = cols_0

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
    plt.savefig(os.path.join(output_dir, f"{fig_name}.png"))
    plt.close()


def scores_percentiles(models_acc, subtitles, metric=None):
    if isinstance(models_acc, pd.DataFrame):
        df = models_acc
    else:
        df = models_acc[metric]
    scores_summary = pd.DataFrame()
    assert len(subtitles) == len(df.columns)
    for col, name in {
        df.columns[i]: subtitles[i] for i in range(len(subtitles))
    }.items():
        metric_summary = pd.DataFrame()
        metric_acc = df[col]
        for model in metric_acc:
            df_model = metric_acc[model]
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
        metric_summary = pd.concat([metric_summary.transpose()], axis=1, keys=[name])
        scores_summary = pd.concat([scores_summary, metric_summary], axis=1)
    if metric is not None:
        scores_summary = pd.concat([scores_summary], keys=[metric])
    return scores_summary


def get_ylim(sharey, bar, scores, regression, models_acc, subtitles, metric=None):
    ylim = None
    if sharey == "row":
        if bar:
            ylim = (
                min([df.min() for df in scores]),
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


def plot_trees_dist(
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
    for drug, models in drugs.items():
        fig_nrows = ceil(len(models.keys()) / fig_ncols)
        fig, axes = plt.subplots(fig_nrows, fig_ncols, sharey=sharey)
        fig_idx = 0
        legend_box = None
        legend_labels = None
        sorted_models = sorted(list(models.keys()))
        for model in sorted_models:
            df = models[model]
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
        plt.subplots_adjust(wspace=wspace, hspace=hspace, top=0.95)

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
