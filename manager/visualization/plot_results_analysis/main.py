import os

import pandas as pd
from manager.data.post_process_results import (
    postprocess_final,
    postprocessing,
    trees_summary,
)
from manager.visualization.plot_results_analysis.plot_results import (
    plot_accuracies,
    plot_averaged,
    plot_splits,
    plot_trees_dist,
)

col_map = {
    "corr_num": "lemonchiffon",
    "corr_num_bias": "khaki",
    "corr_num_sauron": "darkkhaki",
    "corr_num_bias_sauron": "olive",
    "corr_thresh": "paleturquoise",
    "corr_thresh_bias": "mediumturquoise",
    "corr_thresh_sauron": "lightseagreen",
    "corr_thresh_bias_sauron": "teal",
    "all_corr_thresh": "blue",
    "subILP": "thistle",
    "subILP_bias": "plum",
    "subILP_sauron": "violet",
    "subILP_bias_sauron": "purple",
    "targeted_subILP": "lightsteelblue",
    "targeted_subILP_bias": "cornflowerblue",
    "targeted_subILP_sauron": "royalblue",
    "targeted_subILP_bias_sauron": "navy",
    "rf": "peachpuff",
    "all_rf": "darkred",
    "rf_sauron": "sandybrown",
    "random": "lightgrey",
    "random_bias": "grey",
}

if __name__ == "__main__":
    # results_path = "../../../results_larger"
    results_path = "../../../results_average_test"
    output_path = "../../../figures_v4"

    # data uploading kwargs
    regression = True
    targeted = False
    weighted = True
    simple_weight = True
    averaged_results = True
    original = False
    get_trees_info = False
    n_features = 20

    # plot kwargs
    accuracies = True
    per_drug = False
    final = False
    runtime = False
    best_models_count = False
    trees_features = False
    splits = False
    num_features = False

    analysis = {
        "rf_vs_subilp": False,
        "with_sauron": False,
        "targeted": False,
        "data_vs_prior": False,
        "without_bias": False,
        "without_synergy": True,
        "num_features": False,
        "all": False,
    }

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

    condition = f"{dir1}_{dir0}"
    output_dir_ = os.path.join(output_path, dir0, dir1, dir2)
    per_drug_dir = os.path.join(output_dir_, "per_drug")
    os.makedirs(output_dir_, exist_ok=True)

    if regression:
        metrics_ = {
            "overall": "$MSE$",
            "sensitivity": "$MSE_{sens}$",
            "specificity": "$MSE_{res}$",
        }
        thresh = 0.1
        title = f"Regression Performance"
        arranged_metrics = ["sensitivity", "specificity", "overall"]
    else:
        thresh = 0.01
        metrics_ = {
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "f1": "F1-score",
            "youden_j": "Youden's J",
            "mcc": "MCC",  # "Matthews Correlation Coefficient",
        }
        arranged_metrics = ["sensitivity", "specificity", "mcc"]
        title = "Classification Performance"

    if not regression:
        analysis["with_sauron"] = False
        analysis["without_synergy"] = False
    if targeted:
        analysis["rf_vs_subilp"] = False

    row_title_xpos = 0.45
    fig_height = 12
    for key, val in analysis.items():
        if key == "rf_vs_subilp" and val:
            assert not targeted
            if not regression:
                row_title_xpos = 0.5
            fig_name, fig_width = f"rf_vs_subilp_{condition}", 15
            specific_models = [
                "rf",
                "subILP_bias",
                "random_bias",
            ]
        elif key == "with_sauron" and val:
            assert regression
            title = f"{title} - SAURON"
            fig_name, fig_width = f"with_sauron_{condition}", 17
            specific_models = [
                "rf",
                "subILP_bias",
                "random_bias",
                "rf_sauron",
                "subILP_bias_sauron",
            ]
        elif key == "targeted" and val:
            title = f"{title} - Targeted subgraphILP"
            if regression:
                fig_name, fig_width = f"targeted_{condition}", 19
                specific_models = [
                    "rf",
                    "subILP_bias",
                    "random_bias",
                    "rf_sauron",
                    "subILP_bias_sauron",
                    "targeted_subILP_bias_sauron",
                    "targeted_subILP_bias",
                ]
            else:
                row_title_xpos = 0.5
                fig_name, fig_width = f"targeted_{condition}", 15
                specific_models = [
                    "rf",
                    "subILP_bias",
                    "random_bias",
                    "targeted_subILP_bias",
                ]
        elif key == "data_vs_prior" and val:
            if targeted:
                title = f"Targeted {title} - Methods Comparison"
            else:
                title = f"{title} - Methods Comparison"
            if regression:
                fig_name, fig_width = f"data_vs_prior_{condition}", 20
                specific_models = [
                    "rf",
                    "subILP_bias",
                    "random_bias",
                    "rf_sauron",
                    "subILP_bias_sauron",
                    "corr_num_bias",
                    "corr_thresh_bias",
                    "corr_num_bias_sauron",
                    "corr_thresh_bias_sauron",
                ]
                if targeted:
                    fig_name, fig_width = f"targeted_data_vs_prior_{condition}", 22.5
                    specific_models.extend(
                        [
                            "targeted_subILP_bias_sauron",
                            "targeted_subILP_bias",
                        ]
                    )
            else:
                fig_name, fig_width = f"data_vs_prior_{condition}", 15
                specific_models = [
                    "rf",
                    "subILP_bias",
                    "random_bias",
                    "corr_num_bias",
                    "corr_thresh_bias",
                ]
                if targeted:
                    fig_name, fig_width = f"targeted_data_vs_prior_{condition}", 15
                    specific_models.append("targeted_subILP_bias")
        elif key == "num_features" and val:
            fig_height = 7
            fig_name, fig_width = f"num_features", 13
            specific_models = [
                "subILP_bias",
                "corr_num_bias",
                "corr_thresh_bias",
            ]
            if targeted:
                specific_models.append("targeted_subILP_bias")
        elif key == "without_bias" and val:
            title = f"{title} - Bias Ablation"
            if targeted:
                title = f"{title} - Targeted subgraphILP - Bias Ablation"
            fig_name, fig_width = f"without_bias_{condition}", 20
            if regression:
                specific_models = [
                    "random",
                    "random_bias",
                    "subILP",
                    "subILP_bias",
                    "subILP_bias_sauron",
                    "corr_num",
                    "corr_num_bias",
                    "corr_num_bias_sauron",
                    "corr_thresh",
                    "corr_thresh_bias",
                    "corr_thresh_bias_sauron",
                ]
                if targeted:
                    fig_name, fig_width = f"targeted_without_bias_{condition}", 17
                    specific_models.extend(
                        [
                            "targeted_subILP",
                            "targeted_subILP_bias",
                            "targeted_subILP_bias_sauron",
                        ]
                    )
            else:
                fig_width = 15
                specific_models = [
                    "random",
                    "random_bias",
                    "subILP",
                    "subILP_bias",
                    "corr_num",
                    "corr_num_bias",
                    "corr_thresh",
                    "corr_thresh_bias",
                ]
                if targeted:
                    fig_name, fig_width = f"targeted_without_bias_{condition}", 17
                    specific_models.extend(["targeted_subILP", "targeted_subILP_bias"])
        elif key == "without_synergy" and val:
            assert regression
            title = f"{title} - Synergy Ablation"
            if targeted:
                title = f"{title} - Targeted subgraphILP - Synergy Ablation"
            fig_name, fig_width = f"without_synergy_{condition}", 20
            specific_models = [
                "subILP_bias",
                "subILP_sauron",
                "subILP_bias_sauron",
                "corr_num_bias",
                "corr_num_sauron",
                "corr_num_bias_sauron",
                "corr_thresh_bias",
                "corr_thresh_sauron",
                "corr_thresh_bias_sauron",
            ]
            if targeted:
                fig_name, fig_width = f"targeted_without_synergy_{condition}", 19
                specific_models.extend(
                    [
                        "targeted_subILP_bias_sauron",
                        "targeted_subILP_sauron",
                        "targeted_subILP_bias",
                    ]
                )
        elif key == "all" and val:
            fig_name, fig_width = f"performance_{condition}", 21
            if regression:
                specific_models = [
                    "rf",
                    "rf_sauron",
                    "random",
                    "random_bias",
                    "subILP",
                    "subILP_bias",
                    "subILP_sauron",
                    "subILP_bias_sauron",
                    "corr_num",
                    "corr_num_bias",
                    "corr_num_sauron",
                    "corr_num_bias_sauron",
                    "corr_thresh",
                    "corr_thresh_bias",
                    "corr_thresh_sauron",
                    "corr_thresh_bias_sauron",
                ]
                if targeted:
                    fig_width = 29
                    if regression:
                        specific_models.extend(
                            [
                                "targeted_subILP_bias_sauron",
                                "targeted_subILP_sauron",
                                "targeted_subILP_bias",
                                "targeted_subILP",
                            ]
                        )
                    else:
                        specific_models.extend(
                            [
                                "targeted_subILP_bias",
                                "targeted_subILP",
                            ]
                        )
            else:
                fig_name, fig_width = f"performance_{condition}", 17
                specific_models = [
                    "rf",
                    "random",
                    "random_bias",
                    "subILP",
                    "subILP_bias",
                    "corr_num",
                    "corr_num_bias",
                    "corr_thresh",
                    "corr_thresh_bias",
                ]
                if targeted:
                    fig_name, fig_width = f"performance_{condition}", 19
                    specific_models.extend(
                        [
                            "targeted_subILP_bias",
                            "targeted_subILP",
                        ]
                    )
        else:
            if any(analysis.values()):
                continue
            fig_name, fig_width, specific_models = None, None, None

        # results postprocessing
        (
            parameters_grid,
            final_models_acc,
            all_features_acc,
            drugs_acc,
            final_models_parameters,
            final_models_best_features,
            final_models_num_features,
            drug_info,
            metric_best_models,
            metric_best_models_detailed,
            metric_best_models_count,
            parameters_grid_acc,
            parameters_grid_acc_per_drug,
            cross_validation_splits_acc,
            runtimes,
        ) = postprocessing(
            results_path,
            condition,
            targeted,
            regression,
            list(metrics_.keys()),
            thresh,
            specific_models=specific_models,
            all_features=original,
        )
        models_sens_performance = pd.DataFrame(
            metric_best_models["sensitivity"]["test_score"]
        )

        if get_trees_info:
            (
                trees_features_dist,
                trees_features_summary,
                models_features_importance,
            ) = trees_summary(results_path, condition, all_features=original)
            if trees_features:
                plot_trees_dist(
                    trees_features_dist,
                    final_models_num_features,
                    fig_title="Features Distribution Between the Trees",
                    fig_name="trees",
                    output_dir=os.path.join(per_drug_dir, key),
                    col_map=col_map,
                    figsize=(8, 5),
                )

        if specific_models is None:
            break

        if accuracies:
            # fig_name = "performance" if not original else "all_features_performance"
            if original:
                to_arrange = all_features_acc
            else:
                to_arrange = final_models_acc
            final_models_acc_arranged = {m: to_arrange[m] for m in arranged_metrics}

            if averaged_results:
                for metric in final_models_acc_arranged:
                    final_models_acc_arranged[metric] = final_models_acc_arranged[
                        metric
                    ]["test_score"]
                plot_averaged(
                    final_models_acc_arranged,
                    metric_best_models_count,
                    title,
                    fig_name,
                    col_map,
                    output_dir_,
                    metrics_,
                    list(final_models_acc_arranged.keys()),
                    regression=regression,
                    figsize=(fig_width, fig_height),
                    showfliers=False,
                    legend_ncol=1,
                    wspace=0.2,
                    hspace=0.2,
                    row_title_xpos=0.15,
                    row_title_ypos=0.45,
                    fontsize=18,
                    legened_pos_right=0.21,
                    change_right_by=0.12,
                    change_bottom_by=0.0,
                    ax_text_rot=0,
                )
            else:
                plot_accuracies(
                    models_acc=final_models_acc_arranged,
                    output_cv=True,
                    title=title,
                    subtitles=["Test Score", "CV Score"],
                    fig_name=fig_name,
                    metrics=list(final_models_acc_arranged.keys()),
                    col_map=col_map,
                    to_rename=metrics_,
                    output_dir=output_dir_,
                    regression=regression,
                    fig_ncols=2,
                    sharey="row",
                    figsize=(fig_width, fig_height),
                    showfliers=False,
                    legend_ncol=1,
                    legened_pos_right=0.22,
                    change_right_by=0.2,
                    row_title_xpos=0.2,
                    row_title_ypos=-0.4,
                )

        if best_models_count:
            plot_accuracies(
                metric_best_models_count,
                final=False,
                title="Model performance by drug count",
                subtitles=["Test Score", "CV Score"],
                fig_name=f"{fig_name}_best_models_summary",
                metrics=arranged_metrics,
                col_map=col_map,
                to_rename={
                    m: rm.split("-")[1] if "-" in rm else rm
                    for m, rm in metrics_.items()
                },
                output_dir=output_dir_,
                regression=regression,
                ascending=False,
                fig_ncols=2,
                bar=True,
                sharey="row",
                ylabel="number of drugs",
                figsize=(fig_width, fig_height),
                legend_ncol=1,
                legened_pos_right=0.22,
                change_right_by=0.2,
                row_title_xpos=row_title_xpos,  # 0.5, 0.45
                row_title_ypos=0,
                ax_text_rot=0,
            )

        if num_features:
            plot_accuracies(
                final_models_num_features,
                multi_level_df=True,
                final=False,
                title="Number of Features Analysis",
                fig_name=fig_name,
                col_map=col_map,
                output_dir=output_dir_,
                regression=regression,
                ascending=False,
                fig_ncols=2,
                bar=False,
                sharey=False,
                ylabel="number of features",
                figsize=(fig_width, fig_height),
                legend_ncol=1,
                legened_pos_right=0.22,
                change_right_by=0.2,
            )
        if per_drug:
            print(os.path.join(per_drug_dir, key))
            for drug_name, drug_acc in drugs_acc.items():
                drug_acc = {m: drug_acc[m] for m in arranged_metrics}
                plot_accuracies(
                    models_acc=drug_acc,
                    title=f"{drug_name}_accuracies",
                    subtitles=["Test Score", "CV Score"],
                    fig_name=f"{drug_name}_performance",
                    metrics=list(drug_acc.keys()),
                    col_map=col_map,
                    to_rename=metrics_,
                    bar=True,
                    output_dir=os.path.join(per_drug_dir, key),
                    regression=regression,
                    fig_ncols=2,
                    sharey="row",
                    figsize=(fig_width, fig_height),
                    legend_ncol=1,
                    legened_pos_right=0.22,
                    change_right_by=0.2,
                    row_title_xpos=row_title_xpos,  # 0.5, 0.45
                    row_title_ypos=0,
                    ax_text_rot=0,
                )

        if runtime:
            to_rename = {
                "gcv_runtime": "Grid Search Cross Validation"
                if len(parameters_grid) > 1
                else "Cross Validation",
                "model_runtime": "Final Model",
                "rf_train_runtime": "Final Model Fit",
                "rf_test_runtime": "Final Model Predict",
            }
            plot_accuracies(
                models_acc=runtimes,
                single_level_df=True,
                title="Runtime in minutes",
                fig_name="runtimes",
                metrics=[
                    "gcv_runtime",
                    "model_runtime",
                    "rf_train_runtime",
                    "rf_test_runtime",
                ],
                col_map=col_map,
                to_rename=metrics_,
                bar=False,
                output_dir=output_dir_,
                regression=None,
                sharey=False,
                fig_ncols=2,
                figsize=(15, 12),
                legend_ncol=1,
                legened_pos_right=0.22,
                change_right_by=0.2,
                row_title_xpos=0.2,
                row_title_ypos=0.6,
                wspace=0.3,
            )

        if final:
            final_file = "../../results_larger/original_regression_weighted_gt_750/final_model.jsonl"
            final_results = postprocess_final(final_file)
            for k, v in final_results.items():
                df_dict = {col: v[col] for col in arranged_metrics}
                plot_accuracies(
                    df_dict,
                    final=True,
                    title=f"{k}_original",
                    fig_name="final",
                    metrics=arranged_metrics,
                    col_map=col_map,
                    to_rename=metrics_,
                    output_dir=output_dir_,
                    regression=regression,
                    fig_ncols=1,
                    legend_ncol=1,
                    legened_pos_right=0.22,
                    change_right_by=0.2,
                    row_title_xpos=0.1,
                    row_title_ypos=0.6,
                )

        if splits:
            plot_splits(
                cross_validation_splits_acc,
                to_rename=metrics_,
                fig_title="performance per split",
                fig_name="splits",
                fig_ncols=2,
                output_dir=os.path.join(per_drug_dir, key),
                wspace=0.2,
                hspace=0.5,
                figsize=(15, 15),
                fontsize=12,
                legened_pos=(1.5, 7),
                legend_ncol=1,
                sharey=True,
            )

        if not all(analysis.values()):
            break
