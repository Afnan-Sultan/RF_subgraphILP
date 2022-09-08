import os
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd


def plot_with_bar_labels(
    df,
    title,
    bar_label_font,
    xlabel_font,
    output_dir,
    bar_label_color="black",
    transparency=False,
):
    ax = df.plot(kind="bar", figsize=(20, 10), rot=90, title=title)

    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    for container in ax.containers:
        ax.bar_label(
            container, fmt="%.2f", fontsize=bar_label_font, color=bar_label_color
        )
    plt.xticks(fontsize=xlabel_font, color=bar_label_color)
    plt.yticks(color=bar_label_color)
    plt.savefig(
        os.path.join(output_dir, f"{title}.png"),
        bbox_inches="tight",
        transparent=transparency,
    )
    plt.close()


def label_plot(
    ax, title, output_dir, bar_label_font, labels_color, transparency, show=False
):
    ax.set_title(title)
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            fontsize=bar_label_font,
            color=labels_color,
        )
    plt.xticks(color=labels_color)
    plt.yticks(color=labels_color)
    plt.savefig(
        os.path.join(output_dir, f"{title}.png"),
        bbox_inches="tight",
        transparent=transparency,
    )
    if show:
        plt.show()
    plt.close()


def plot_df_stats(
    grouped_df,
    title,
    bar_label_font,
    xlabel_font,
    output_dir,
    labels_color,
    transparency,
):
    mean_df = grouped_df.mean()
    median_df = grouped_df.median()
    plot_with_bar_labels(
        mean_df,
        f"mean {title}",
        bar_label_font,
        xlabel_font,
        output_dir,
        bar_label_color=labels_color,
        transparency=transparency,
    )
    plot_with_bar_labels(
        median_df,
        f"median {title}",
        bar_label_font,
        xlabel_font,
        output_dir,
        bar_label_color=labels_color,
        transparency=transparency,
    )


def get_ax_legends(fig):
    labels_lists = [ax.get_legend_handles_labels()[1] for ax in fig.axes]
    labels = []
    for ll in labels_lists:
        for label in ll:
            if label not in labels:
                labels.append(label)
    return labels


def plot_subplots(
    to_plot,
    to_enumerate,
    title,
    figure_name,
    output_dir,
    to_plot_to_dict=False,
    ax_prefix=None,
    xlabel="",
    ylabel="",
    figsize=(10, 10),
    hspace=0.5,
    transparent=False,
    loc="lower right",
    legend_fontsize="large",
):
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=hspace)
    plt.suptitle(title)
    for idx, plotted in enumerate(to_enumerate):
        ax = fig.add_subplot(ceil(len(to_enumerate) / 2), 2, idx + 1)
        if to_plot_to_dict:
            pd.DataFrame(to_plot[plotted].to_dict()).plot(ax=ax)
        else:
            pd.DataFrame(to_plot[plotted]).plot(ax=ax)
        if ax_prefix is not None:
            ax.set_title(f"{ax_prefix}_{plotted}")
        else:
            ax.set_title(plotted)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.get_legend().remove()
    labels = get_ax_legends(fig)
    fig.legend(labels, loc=loc, fontsize=legend_fontsize)
    plt.savefig(
        os.path.join(output_dir, f"{figure_name}.png"),
        bbox_inches="tight",
        transparent=transparent,
    )
    plt.close()
