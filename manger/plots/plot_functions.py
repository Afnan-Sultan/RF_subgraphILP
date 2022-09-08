import os
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd
from manger.plots.plot_helpers import (
    get_ax_legends,
    label_plot,
    plot_df_stats,
    plot_subplots,
    plot_with_bar_labels,
)
from manger.plots.plot_leveled import plot_levels
from manger.plots.process_results import serialize_dict


def plot_runtime(df, title, bar_label_font=10, xlabel_font=15, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    plot_with_bar_labels(
        df["gcv_runtime"].groupby(level=0).mean(),
        f"mean grid search cross validation {title}",
        bar_label_font,
        xlabel_font,
        output_dir,
    )
    plot_with_bar_labels(
        df["model_runtime"].groupby(level=0).mean(),
        f"mean best model {title}",
        bar_label_font,
        xlabel_font,
        output_dir,
    )

    grouped_df = df.drop(["gcv_runtime", "model_runtime"], axis=1).groupby(level=0)
    mean_df = grouped_df.mean()
    plot_with_bar_labels(
        mean_df, f"mean Random Forest {title}", bar_label_font, xlabel_font, output_dir
    )


def plot_acc(
    df_to_plot,
    subset,
    title,
    plot_all=True,
    plot_single_drugs=True,
    drop_cols=None,
    output_dir="figures",
    figsize=(20, 10),
    bar_label_font=10,
    xlabel_font=15,
    drug_name=None,
    labels_color="black",
    transparency=False,
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.json_normalize(df_to_plot)
    df.columns = df.columns.str.split(".").map(tuple)

    if drop_cols is not None:
        df.drop(
            drop_cols if isinstance(drop_cols, list) else [drop_cols],
            axis=1,
            level=1,
            inplace=True,
        )

    # prepare a plot for each drug, only when a subset is specified for time's sake
    if subset is not None:
        new_df = df.stack([2, 0, 1]).reset_index(0, drop=True)
        new_df = new_df.loc[subset]

        if drug_name is not None:  # to plot specific drug only without saving
            new_df.loc[drug_name].plot(kind="bar", figsize=figsize)
            plt.title(f"{title} - {drug_name}")
            # plt.show()
            plt.close()
        elif plot_single_drugs:  # to plot all drugs individually
            drug_out_dir = os.path.join(output_dir, f"{subset}_per_drug")
            for drug in set(new_df.index.get_level_values(0)):
                os.makedirs(drug_out_dir, exist_ok=True)

                ax = new_df.loc[drug].plot(kind="bar", figsize=figsize)
                label_plot(
                    ax,
                    f"{title} - {drug}",
                    drug_out_dir,
                    bar_label_font,
                    labels_color,
                    transparency,
                )
    else:
        new_df = df.stack([0, 1, 2]).reset_index(0, drop=True)

    if plot_all:
        plot_levels(new_df, f"{title}_drugs", output_dir, add_vals=False, lypos=-0.3)

    grouped_df = new_df.groupby(level=1)
    drugs_per_model = grouped_df.size()
    # TODO: this assumes only one model has less drugs
    models_to_drop = [
        model for model, count in drugs_per_model.items() if count != len(df_to_plot)
    ]
    if len(models_to_drop) > 0:
        to_drop = models_to_drop[0]
        grouped_models_df = new_df.drop(grouped_df.get_group(to_drop).index).groupby(
            level=1
        )
        plot_df_stats(
            grouped_models_df,
            f"{title} - without {to_drop}",
            bar_label_font,
            xlabel_font,
            output_dir,
            labels_color,
            transparency,
        )

        drugs_to_plot = [x[0] for x in list(grouped_df.groups[to_drop])]
        if len(drugs_to_plot) > 1:
            grouped_drugs_df = new_df.loc[drugs_to_plot].groupby(level=1)
            plot_df_stats(
                grouped_drugs_df,
                f"{title} -  {len(drugs_to_plot)} drugs",
                bar_label_font,
                xlabel_font,
                output_dir,
                labels_color,
                transparency,
            )
    else:
        plot_df_stats(
            grouped_df,
            title,
            bar_label_font,
            xlabel_font,
            output_dir,
            labels_color,
            transparency,
        )


def plot_param_selection(
    df,
    output_dir="figures",
    transparent=False,
    rot=90,
    figsize=(15, 5),
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    hspace=0.5,
):
    os.makedirs(output_dir, exist_ok=True)

    num_features = None
    dfs_to_plot = {}
    if "num_features" in df.columns:
        new_title = "number of features per model"
        num_features = df["num_features"].drop("corr_num_rf", axis=1)
        if "original" in num_features.columns:
            num_features.drop("original", axis=1, inplace=True)
        elif "rf" in num_features.columns:
            num_features.drop("rf", axis=1, inplace=True)
        num_features.plot(kind="bar", rot=rot, figsize=figsize, title=new_title)
        plt.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
        plt.savefig(
            os.path.join(output_dir, f"{new_title}.png"),
            bbox_inches="tight",
            transparent=transparent,
        )
        plt.close()

        df = df.drop("num_features", axis=1)

    params = set([name[0] for name in df.columns])
    plt.subplots_adjust(hspace=hspace)
    for fig_idx, param in enumerate(params):
        if param != "num_features":
            param_df = df[param]
            df_count = pd.DataFrame(
                columns=param_df.columns,
                index=set(list(param_df.dropna().values.flatten())),
            )
            for model in param_df:
                model_df = pd.DataFrame(param_df[model].value_counts(dropna=True))
                for idx in model_df.index:
                    df_count.loc[idx, model] = model_df.loc[idx].to_list()[0]
            ax = plt.subplot(len(params) // 2, 2, fig_idx + 1)
            to_plot = df_count.transpose()
            dfs_to_plot[param] = to_plot
            to_plot.plot(kind="bar", title=f"{param}", ax=ax, figsize=(15, 5))
    plt.savefig(
        os.path.join(output_dir, f"params.png"),
        bbox_inches="tight",
        transparent=transparent,
    )
    plt.close()
    return num_features, dfs_to_plot


def plot_param_train_error(
    params_acc,
    score_dict,
    title="parameter combination train score",
    output_dir="figures",
    figsize=(20, 10),
    transparent=False,
    loc="lower right",
    legend_fontsize="large",
    hspace=0.5,
):
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=hspace)
    plt.suptitle(f"{title} per model ", fontsize=20)
    for idx, subset in enumerate(list(params_acc.keys())):
        # plot average drug performance for each parameter, all models in one plot
        ax = fig.add_subplot(ceil(len(params_acc) / 2), 2, idx + 1)
        df = params_acc[subset].sort_index().groupby(level=0).mean()
        df.index = df.index.astype(int)
        df.sort_index().plot(ax=ax)

        ax.set_xlabel("hyperparameters index")
        ax.set_ylabel(f"average {score_dict[subset]}")
        ax.get_legend().remove()
    labels = get_ax_legends(fig)
    fig.legend(labels, loc=loc, fontsize=legend_fontsize)
    plt.savefig(
        os.path.join(output_dir, f"{title} per model.png"),
        bbox_inches="tight",
        transparent=transparent,
    )
    plt.close()


def plot_params_acc(
    param_acc_subset,
    params_acc_per_drug,
    params_df,
    score,
    title="parameter combination train score",
    output_dir="figures",
    tuning_per_model=True,
    param_per_drug=True,
    figsize=(20, 15),
    transparent=False,
    loc="lower right",
    legend_fontsize="large",
    hspace=1.5,
):
    os.makedirs(output_dir, exist_ok=True)
    df = param_acc_subset.sort_index().groupby(level=0).mean()
    df.index = df.index.astype(int)

    if tuning_per_model:
        max_features = params_df.groupby("max_features").groups
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=hspace)
        plt.suptitle(f"{score} - parameter tuning per model", fontsize=20)
        for idx, model in enumerate(df.columns):
            ax = fig.add_subplot(ceil(len(df.columns) / 2), 2, idx + 1)
            new_df = pd.DataFrame()
            for max_feat, indices in max_features.items():
                new_df[max_feat] = df[model].iloc[list(indices)].to_list()
            new_df.index = params_df["min_samples_leaf"].iloc[new_df.index]
            new_df.plot(ax=ax)
            ax.set_title(model)
            ax.set_xlabel("min_samples_leaf")
            ax.set_ylabel(f"average {score}")
        #     ax.get_legend().remove()
        # labels = get_ax_legends(fig)
        # fig.legend(labels, loc=loc, fontsize=legend_fontsize)
        plt.savefig(
            os.path.join(output_dir, f"parameter tuning per model {score}.png"),
            bbox_inches="tight",
            transparent=transparent,
        )
        plt.close()

    if param_per_drug:
        # plot drug performance per parameters, all models in one figure
        params_acc_per_drug.dropna(inplace=True)
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=hspace)
        plt.suptitle(f"{title} per drug", fontsize=20)
        for idx, model in enumerate(params_acc_per_drug.columns):
            ax = fig.add_subplot(ceil(len(params_acc_per_drug.columns) / 2), 2, idx + 1)
            pd.DataFrame(
                params_acc_per_drug[model].to_list(), index=params_acc_per_drug.index
            ).transpose().plot(ax=ax)
            ax.set_title(model)
            ax.set_xlabel("hyperparameters index")
            ax.set_ylabel(f"average {score}")
            ax.get_legend().remove()
        labels = get_ax_legends(fig)
        fig.legend(labels, loc=loc, fontsize=legend_fontsize)
        plt.savefig(
            os.path.join(output_dir, f"{title} per drug.png"),
            bbox_inches="tight",
            transparent=transparent,
        )
        plt.close()


def plot_splits(
    splits_dict,
    title,
    sub_metric,
    output_dir,
    figsize=(10, 10),
    transparent=False,
    loc="lower right",
    legend_fontsize="large",
    hspace=0.5,
):
    os.makedirs(output_dir, exist_ok=True)
    for drug, rf_models in splits_dict.items():
        plot_subplots(
            rf_models,
            list(rf_models.keys()),
            f"{title} for {drug}",
            f"{title} - {drug} - per model",
            output_dir,
            xlabel="hyperparameters index",
            ylabel=sub_metric,
            figsize=figsize,
            hspace=hspace,
            transparent=transparent,
            loc=loc,
            legend_fontsize=legend_fontsize,
        )

    df = serialize_dict(splits_dict, [2, 1]).transpose()
    for drug in df.index:
        drug_df = pd.DataFrame(df.loc[drug]).unstack([0]).droplevel(0, axis=1)

        plot_subplots(
            drug_df,
            drug_df.columns.to_list(),
            f"{title} for {drug}",
            f"{title} - {drug} - per split",
            output_dir,
            to_plot_to_dict=True,
            ax_prefix="cv",
            xlabel="hyperparameters index",
            ylabel=sub_metric,
            figsize=figsize,
            hspace=hspace,
            transparent=transparent,
            loc=loc,
            legend_fontsize=legend_fontsize,
        )


def plot_features_importance(
    features_dfs,
    n_features,
    output_dir,
    figsize1=(30, 15),
    figsize2=(30, 15),
    transparent=False,
    loc="lower right",
    legend_fontsize="large",
    xrot=45,
    hspace=1.2,
):
    os.makedirs(output_dir, exist_ok=True)
    for drug in features_dfs.index:
        df_list = []
        drug_df = features_dfs.loc[drug].dropna()
        for model in drug_df.index:
            features_df = drug_df[model]
            first_n_features = features_df[:n_features]
            df_list.append(first_n_features)
        # common = set(first_n_features_ilp["genes"].to_list()).intersection(first_n_features_orig["genes"].to_list()

        plt.figure(figsize=figsize1)
        plt.subplots_adjust(hspace=hspace)
        plt.suptitle(
            f"first {n_features} best features for {drug}"
        )  # \nIntersection: {common}', y=1.05)

        for idx, model in enumerate(drug_df.index):
            ax = plt.subplot(ceil(len(drug_df)), 2, idx + 1)
            temp_df = df_list[idx].copy(deep=True)
            temp_df.iloc[:, 0] = temp_df.iloc[:, 0].astype(float)
            temp_df.plot(x="GeneSymbol", ax=ax, kind="bar", figsize=figsize2, rot=xrot)
            ax.set_title(model)

        plt.savefig(
            os.path.join(output_dir, f"best features for {drug}.png"),
            bbox_inches="tight",
            transparent=transparent,
        )
        plt.close()
