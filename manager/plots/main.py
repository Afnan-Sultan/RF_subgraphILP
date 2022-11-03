import os

from manager.data.post_process_results import (
    postprocess_final,
    postprocessing,
    trees_summary,
)
from manager.visualization.plot_results import plot_accuracies, plot_trees_dist

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
    results_path = "../../results_larger"
    output_path = "../../figures_v3"

    regression = True
    weighted = True
    simple_weight = True
    targeted = False
    original = True
    n_features = 20

    runtime = False
    per_drug = False
    accuracies = True
    final = False
    best_models_count = False
    trees_features = False
    splits = False

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
            "overall": "MSE - Overall",
            "sensitivity": "MSE - Sensitive",
            "specificity": "MSE - Resistant",
        }
        thresh = 0.1
        title = f"MSE Scores - {condition}"
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
        title = condition

    # results postprocessing
    (
        parameters_grid,
        final_models_acc,
        all_features_acc,
        drugs_acc,
        final_models_parameters,
        final_models_best_features,
        final_models_num_features,
        metric_best_models,
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
        all_features=original,
    )

    if best_models_count:
        plot_accuracies(
            metric_best_models_count,
            final=True,
            title="Best Models' drugs count",
            subtitles=["Number of drugs"],
            fig_name="best_models_summary",
            metrics=arranged_metrics,
            col_map=col_map,
            to_rename=metrics_,
            output_dir=output_dir_,
            regression=regression,
            ascending=False,
            fig_ncols=1,
            bar=True,
            sharey=False,
            figsize=(15, 12),
            legend_ncol=1,
            legened_pos_right=0.22,
            change_right_by=0.2,
            row_title_xpos=0.1,
            row_title_ypos=0.6,
        )

    if per_drug:
        for drug_name, drug_acc in drugs_acc.items():
            drug_acc = {m: drug_acc[m] for m in arranged_metrics}
            plot_accuracies(
                models_acc=drug_acc,
                title=f"{drug_name}_accuracies",
                fig_name=f"{drug_name}_performance",
                metrics=list(drug_acc.keys()),
                col_map=col_map,
                to_rename=metrics_,
                bar=True,
                output_dir=per_drug_dir,
                regression=regression,
                fig_ncols=2,
                sharey="row",
                figsize=(15, 12),
                legend_ncol=1,
                legened_pos_right=0.22,
                change_right_by=0.2,
                row_title_xpos=0.2,
                row_title_ypos=0.6,
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
            title="Runtime in minutes",
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

    if accuracies:
        if original:
            to_arrange = all_features_acc
        else:
            to_arrange = final_models_acc
        final_models_acc_arranged = {m: to_arrange[m] for m in arranged_metrics}
        plot_accuracies(
            models_acc=final_models_acc_arranged,
            title=title,
            fig_name="performance" if not original else "all_features_performance",
            metrics=list(final_models_acc_arranged.keys()),
            col_map=col_map,
            to_rename=metrics_,
            output_dir=output_dir_,
            regression=regression,
            fig_ncols=2,
            sharey="row",
            figsize=(15, 12),
            showfliers=False,
            legend_ncol=1,
            legened_pos_right=0.22,
            change_right_by=0.2,
            row_title_xpos=0.2,
            row_title_ypos=0.6,
        )

    if final:
        final_file = (
            "../../results_larger/original_regression_weighted_gt_750/final_model.jsonl"
        )
        final_results = postprocess_final(final_file)
        for k, val in final_results.items():
            df_dict = {col: val[col] for col in arranged_metrics}
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

    if trees_features:
        trees_features_dist, trees_features_summary = trees_summary(
            results_path, condition, all_features=original
        )
        for ml, drugs in trees_features_dist.items():
            if len(drugs[list(drugs.keys())[0]]) == 0:
                continue
            plot_trees_dist(
                drugs,
                fig_title="count of trees with shared features - biased RF",
                fig_name="trees",
                fig_ncols=2,
                output_dir=per_drug_dir,
                wspace=0.2,
                hspace=0.5,
                figsize=(15, 10),
                xlabel="number of features",
            )

    if splits:
        plot_trees_dist(
            cross_validation_splits_acc,
            to_rename=metrics_,
            fig_title="performance per split",
            fig_name="splits",
            fig_ncols=2,
            output_dir=per_drug_dir,
            wspace=0.2,
            hspace=0.5,
            figsize=(15, 15),
            fontsize=12,
            legened_pos=(1.5, 7),
            legend_ncol=1,
            sharey=True,
        )
