import os

from manager.data.post_process_results import (
    comp_features_intersection,
    compare_bias_ablation_features,
    get_data_vs_prior_features_info,
    postprocess_final,
    postprocessing,
    recruit_drugs,
    trees_summary,
)
from manager.visualization.plot_results_analysis.plot_results import (
    plot_accuracies,
    plot_averaged,
    plot_common_features_mat,
    plot_params_acc,
    plot_single_boxplot,
    plot_splits,
    plot_trees_dist,
)
from manager.visualization.plot_results_analysis.variables import (
    analysis_utils,
    col_map,
    get_metrics,
    output_mkdir,
)
from tqdm import tqdm

if __name__ == "__main__":
    results_path = "../../../results_average_test"
    # results_path = "../../../results_cv"
    output_path = "../../../figures_v4"
    # output_path = "../../../figures_v4/cv"

    # data uploading kwargs
    regression = True
    targeted = False
    weighted = True
    simple_weight = True
    averaged_results = True if "average" in results_path else False
    original = False

    # postprocessing kwargs
    get_trees_info = False
    get_features_intersection = True
    n_features = 100
    compare_bias_ablation = True

    # plot kwargs
    accuracies = False
    per_drug = False
    final = False
    runtime = True
    best_models_count = False
    plot_features_dist = False
    splits = False
    num_features = False
    plot_intersection_matrix = False
    plot_parameters_performance = False

    analysis = {
        "rf_vs_subilp": 0,
        "with_sauron": 0,
        "targeted": 0,
        "data_vs_prior": 0,
        "without_bias": 0,
        "without_synergy": 0,
        "num_features": 0,
        "all": 0,
    }

    condition, output_dir, per_drug_dir = output_mkdir(
        weighted, simple_weight, regression, targeted, output_path
    )

    renamed_metrics, title, arranged_metrics = get_metrics(regression)

    if not regression:
        analysis["with_sauron"] = False
        analysis["without_synergy"] = False
    if targeted:
        analysis["rf_vs_subilp"] = False

    analysis_list = [key for key, val in analysis.items() if val]
    analysis_params = analysis_utils(
        analysis_list, regression, targeted, condition, title
    )

    parameters_grid = None
    final_models_acc = None
    all_features_acc = None
    drugs_acc = None
    final_models_parameters = None
    final_models_best_features = None
    final_models_num_features = None
    drug_info = None
    metric_best_models = None
    metric_best_models_count = None
    parameters_grid_acc = None
    parameters_grid_acc_per_drug = None
    cross_validation_splits_acc = None
    runtimes = None

    for analysis_type, params in analysis_params:
        if not averaged_results:
            if accuracies:
                specific_models = [
                    "corr_thresh_bias",
                    "corr_thresh_bias_tuned",
                    "subILP_bias",
                    "subILP_bias_tuned",
                ]
            else:
                specific_models = [
                    "corr_thresh_bias_tuned",
                    "subILP_bias_tuned",
                ]
        else:
            specific_models = params["specific_models"]
        title = params["title"]
        fig_name = params["fig_name"]
        fig_height = params["fig_height"]
        fig_width = params["fig_width"]
        row_title_xpos = params["row_title_xpos"]

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
            metric_best_models_count,
            parameters_grid_acc,
            parameters_grid_acc_per_drug,
            cross_validation_splits_acc,
            runtimes,
        ) = postprocessing(
            results_path=results_path,
            condition=condition,
            targeted=targeted,
            regression=regression,
            metrics=list(renamed_metrics.keys()),
            averaged_results=averaged_results,
            specific_models=specific_models,
            all_features=original,
            output_dir=output_dir,
            csv_name=analysis_type,
        )
        recruited_drugs = recruit_drugs(metric_best_models)
        # x = final_models_acc["sensitivity"]["test_score"].loc[recruited_drugs["corr_thresh_bias_sauron"], :]

        if accuracies:
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
                    acc_dict=final_models_acc_arranged,
                    by_count_dict=metric_best_models_count,
                    title=title,
                    fig_name=fig_name,
                    col_map=col_map,
                    output_dir=output_dir,
                    to_rename=renamed_metrics,
                    metrics=list(final_models_acc_arranged.keys()),
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
                    to_rename=renamed_metrics,
                    output_dir=output_dir,
                    regression=regression,
                    fig_ncols=2,
                    sharey="row",
                    figsize=(fig_width, fig_height),
                    showfliers=False,
                    legend_ncol=1,
                    legened_pos_right=0.22,
                    change_right_by=0.2,
                    row_title_xpos=0.2,
                    row_title_ypos=-0.05,
                    ax_text_rot=0,
                )

        if best_models_count:
            plot_accuracies(
                models_acc=metric_best_models_count,
                final=False,
                title="Model performance by drug count",
                subtitles=["Test Score", "CV Score"],
                fig_name=f"{fig_name}_best_models_summary",
                metrics=arranged_metrics,
                col_map=col_map,
                to_rename={
                    m: rm.split("-")[1] if "-" in rm else rm
                    for m, rm in renamed_metrics.items()
                },
                output_dir=output_dir,
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

        if analysis_type == "num_features" and num_features:
            plot_accuracies(
                models_acc=final_models_num_features,
                multi_level_df=True,
                final=False,
                title="Number of Features Analysis",
                fig_name=fig_name,
                col_map=col_map,
                output_dir=output_dir,
                regression=regression,
                ascending=False,
                fig_ncols=2,
                bar=False,
                sharey=False,
                ylabel="number of features",
                figsize=(fig_width, fig_height),
                wspace=0.2,
                legend_ncol=1,
                legened_pos_right=0.22,
                change_right_by=0.2,
            )

        if per_drug:
            print(os.path.join(per_drug_dir, analysis_type))
            for drug_name, drug_acc in tqdm(
                drugs_acc.items(), desc="plotting performance per drug..."
            ):
                drug_acc = {m: drug_acc[m] for m in arranged_metrics}
                plot_accuracies(
                    models_acc=drug_acc,
                    title=f"{drug_name}_accuracies",
                    subtitles=["Test Score", "CV Score"],
                    fig_name=f"{drug_name}_performance",
                    metrics=list(drug_acc.keys()),
                    col_map=col_map,
                    to_rename=renamed_metrics,
                    bar=True,
                    averaged_results=averaged_results,
                    output_dir=os.path.join(per_drug_dir, analysis_type),
                    regression=regression,
                    fig_ncols=2,
                    sharey="row",
                    figsize=(fig_width, fig_height),
                    legend_ncol=1,
                    legened_pos_right=0.22,
                    change_right_by=0.2,
                    row_title_xpos=0.2,  # row_title_xpos,  # 0.5, 0.45
                    row_title_ypos=0.2,
                    ax_text_rot=0,
                )

        if runtime:
            to_rename = {
                "gcv_runtime": "5-Fold Runs"
                if averaged_results
                else "Grid Search Cross Validation"
                if len(parameters_grid) > 1
                else "Cross Validation",
                "model_runtime": "Average Single Model"
                if averaged_results
                else "Final Model",
                "rf_train_runtime": "Average Model Fit"
                if averaged_results
                else "Final Model Fit",
                "rf_test_runtime": "Average Model Predict"
                if averaged_results
                else "Final Model Predict",
            }
            if regression:
                title = "Regression - Runtime in Minutes"
                legend_pos = 0.28 if targeted else 0.25
                figsize = (15, 12)
            else:
                title = "Classification - Runtime in Minutes"
                legend_pos = 0.28 if targeted else 0.24
                figsize = (12, 12)
            plot_accuracies(
                models_acc=runtimes,
                single_level_df=True,
                in_minutes=True,
                title=title,
                fig_name="runtimes",
                metrics=[
                    "gcv_runtime",
                    "model_runtime",
                    "rf_train_runtime",
                    "rf_test_runtime",
                ],
                col_map=col_map,
                to_rename=to_rename,
                bar=False,
                output_dir=output_dir,
                regression=None,
                sharey=False,
                fig_ncols=2,
                figsize=figsize,
                legend_ncol=1,
                legened_pos_right=0.28 if targeted else 0.25,
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
                    models_acc=df_dict,
                    final=True,
                    title=f"{k}_original",
                    fig_name="final",
                    metrics=arranged_metrics,
                    col_map=col_map,
                    to_rename=renamed_metrics,
                    output_dir=output_dir,
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
                drugs=cross_validation_splits_acc,
                to_rename=renamed_metrics,
                fig_title="performance per Fold",
                fig_name="splits",
                fig_ncols=2 if regression else 3,
                output_dir=os.path.join(per_drug_dir, analysis_type),
                wspace=0.2,
                hspace=0.5,
                figsize=(12, 10) if regression else (17, 5),
                fontsize=12,
                legened_pos=(1.5, 2.5),
                legend_ncol=1,
                sharey=True,
            )

        if plot_parameters_performance:
            parameters_grid_acc = {m: parameters_grid_acc[m] for m in arranged_metrics}
            plot_params_acc(
                parameters_grid_acc,
                parameters_grid,
                output_dir,
                renamed_metrics,
            )

    if regression:
        trees_folder = "../../../results_average_test/regression_weighted_biased_gt_750/rf_trees_info"
        files_dir = "../../../results_average_test/regression_weighted_gt_750"
        biased_file = "../../../results_average_test/regression_weighted_biased_gt_750/trees_features_summary.json"
    else:
        trees_folder = "../../../results_average_test/classification_weighted_biased_gt_750/rf_trees_info"
        files_dir = "../../../results_average_test/classification_weighted_gt_750"
        biased_file = "../../../results_average_test/classification_weighted_biased_gt_750/trees_features_summary.json"

    if get_trees_info:
        (
            trees_features_dist,
            trees_features_summary,
            models_features_importance,
        ) = trees_summary(trees_folder=trees_folder, all_features=original)
        if get_features_intersection:
            to_compare = ["subgraphilp", "corr_num", "corr_thresh"]
            important = comp_features_intersection(
                models_features_importance,
                to_compare,
                n_features,
                output_dir,
                add_bias=True,
            )
            features, _ = get_data_vs_prior_features_info(biased_file)
            selected = comp_features_intersection(features, to_compare, selected=True)
            if plot_intersection_matrix:
                plot_dir = os.path.join(per_drug_dir, "trees")
                if regression:
                    name = "Regression - Prior Vs Statistical"
                else:
                    name = "Classification - Prior Vs Statistical"
                for drug, info in important.items():
                    df = info["num_common"]
                    plot_common_features_mat(
                        df, drug, plot_dir, n_features=n_features, name=name
                    )

                    df = selected[drug]["num_common"]
                    plot_common_features_mat(
                        df, drug, plot_dir, name=name, selected=True
                    )
        if plot_features_dist:
            plot_trees_dist(
                drugs=trees_features_dist,
                fig_title="Features Summary Between the Trees",
                fig_name="trees",
                output_dir=os.path.join(per_drug_dir, "trees"),
                col_map=col_map,
                figsize=(8, 5),
            )

    if compare_bias_ablation:
        corr_file = os.path.join(files_dir, "initial_features_info.jsonl")
        subilp_file = os.path.join(files_dir, "initial_features_info_subgraph.json")
        (
            ablation_features,
            ablation_num_features,
            ablation_important_features,
        ) = compare_bias_ablation_features(
            results_path,
            corr_file,
            subilp_file,
            biased_file,
            regression,
            list(renamed_metrics.keys()),
            condition,
            targeted,
            averaged_results,
            original,
            output_dir,
        )
        to_compare = [
            "subILP",
            "subILP_bias",
            "corr_num",
            "corr_num_bias",
            "corr_thresh",
            "corr_thresh_bias",
        ]
        plot_single_boxplot(
            ablation_num_features,
            col_map,
            to_compare,
            output_dir,
            fig_name="Bias ablation - num features",
            figsize=(12, 10),
        )

        selected_features_intersection = comp_features_intersection(
            ablation_features, to_compare, selected=True
        )

        if regression:
            name = "Regression - Bias Ablation"
        else:
            name = "Classification - Bias Ablation"
        if plot_intersection_matrix:
            plot_dir = os.path.join(per_drug_dir, "bias_ablation")
            for drug, info in selected_features_intersection.items():
                df = info["num_common"]
                plot_common_features_mat(
                    df,
                    drug,
                    plot_dir,
                    name=name,
                    figsize=(10, 10),
                    selected=True,
                    title_pos=0.95,
                )

        important_features_intersection = comp_features_intersection(
            ablation_important_features,
            to_compare,
            n_features,
            output_dir,
        )
        if plot_intersection_matrix:
            plot_dir = os.path.join(per_drug_dir, "bias_ablation")
            for drug, info in important_features_intersection.items():
                df = info["num_common"]
                plot_common_features_mat(
                    df,
                    drug,
                    plot_dir,
                    n_features=n_features,
                    figsize=(10, 10),
                    name=name,
                    title_pos=0.95,
                )
