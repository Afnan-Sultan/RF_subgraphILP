import os
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np


def plot_clustered_imposed(
    dfall, labels=None, title="multiple stacked bar plot", H="/", **kwargs
):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe
    code from here:
    https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall:  # for each data frame
        axe = df.plot(
            kind="bar",
            linewidth=1,
            stacked=False,
            ax=axe,
            legend=True,
            figsize=(20, 10),
            width=0,
            edgecolor="black",
            alpha=0.5,
            **kwargs,
        )  # make bar plots

    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col + 1, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i : i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))  # edited part
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.0)
    axe.set_xticklabels(dfall[0].index, rotation=90)
    axe.set_title(title)

    # Add invisible data to add another legend
    n = []
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1])
    axe.add_artist(l1)
    axe.get_figure().tight_layout()
    plt.savefig(f'figures/{title.split("for")[0]}.jpeg')
    return axe


def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [0, ypos], transform=ax.transAxes, color="gray")
    line.set_clip_on(False)
    ax.add_line(line)


def label_len(my_index, level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k, g in groupby(labels)]


def label_group_bar_table(ax, df, rotation=90, fontsize=20, lypos=0):
    scale = 1.0 / df.index.size  # x-labels positions with respect to the x-axis bars
    h_start = 0  # start of shading
    for level in range(df.index.nlevels)[::-1]:
        pos = 0  # start of boundary lines
        for label, rpos in label_len(df.index, level):
            lxpos = (
                pos + 0.5 * rpos
            ) * scale  # start and end of x-label position on the x-axis
            ax.text(
                lxpos,
                lypos,
                label,
                ha="center",
                transform=ax.transAxes,
                rotation=rotation,
                fontsize=fontsize,
            )
            add_line(ax, pos * scale, lypos)
            pos += rpos
            if level == 0:
                # add disjointed shadows to subsets of the plot
                plt.axvspan(h_start - 0.4, pos - 0.6, facecolor="gray", alpha=0.2)
                h_start = pos
        add_line(ax, pos * scale, lypos)
        lypos -= 0.35


def plot_levels(
    df,
    title,
    output_dir,
    add_vals=True,
    figsize=(30, 15),
    rotation=90,
    fontsize=20,
    lypos=-0.2,
    bar_label_font=10,
    transparency=False,
):

    ax = df.plot(kind="bar", figsize=figsize, rot=90)
    ax.set_xticklabels("")
    ax.set_xlabel("")
    label_group_bar_table(ax, df, rotation=rotation, fontsize=fontsize, lypos=lypos)
    ax.set_title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    if add_vals:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=bar_label_font)
    plt.savefig(
        os.path.join(output_dir, f"{title}.png"),
        bbox_inches="tight",
        transparent=transparency,
    )
    plt.close()
