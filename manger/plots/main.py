import os

from plot_functions import (
    plot_acc,
    plot_features_importance,
    plot_param_selection,
    plot_param_train_error,
    plot_params_acc,
    plot_runtime,
    plot_splits,
)
from process_results import acc_runtime, aggregate_result_files, best_est_info


def plot_drugs_acc_subsets(
    regression,
    model_acc,
    params_acc,
    params_acc_per_drug,
    params_df,
    splits_acc,
    output_path,
    param_train_error=True,
    best_model_acc=True,
    parameters_acc=True,
    splits=True,
    hspace=0.8,
):
    ml_subsets = {
        "regression": {
            "overall": "MSE scores",
            "sensitivity": "MSE scores - resistant",
            "specificity": "MSE scores - sensitive",
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
        subsets = ml_subsets["regression"]
    else:
        subsets = ml_subsets["classification"]
    if param_train_error:
        plot_param_train_error(
            params_acc,
            subsets,
            output_dir=os.path.join(
                output_path,
                "parameters",
            ),
        )

    for subset, title in subsets.items():
        if best_model_acc:
            plot_acc(
                model_acc,
                subset,
                title,
                plot_all=False,
                output_dir=os.path.join(output_path, "best_model_performance"),
            )
        if parameters_acc:
            plot_params_acc(
                params_acc[subset],
                params_acc_per_drug[subset],
                params_df,
                title,
                output_dir=os.path.join(
                    output_path,
                    "parameters",
                ),
                hspace=hspace,
            )
        if splits:
            plot_splits(
                splits_acc[subset],
                title,
                subset,
                output_dir=os.path.join(
                    output_path,
                    "splits",
                ),
            )


if __name__ == "__main__":
    results_path_ = "../../results"

    weighted = False
    simple_weight = True
    regression_ = False
    targeted_ = False
    condition_ = "classification_notweighted"
    n_features = 20
    hspace = 1.2

    if weighted and simple_weight:
        dir0 = "weighted_simple"
    elif weighted and not simple_weight:
        dir0 = "weighted_linear"
    else:
        dir0 = "not_weighted"
    if regression_:
        dir1 = "regression"
    else:
        dir1 = "classification"
    if targeted_:
        dir2 = "targeted"
    else:
        dir2 = "not_targeted"
    output_dir = os.path.join("..", "..", "figures", dir0, dir1, dir2)

    # load results
    drugs_dict_ = aggregate_result_files(results_path_, condition_, targeted_)

    # extract runtime and accuracies information
    (
        grid_params_,
        best_model_acc_,
        all_params_acc_,
        params_acc_per_drug_,
        all_splits_acc_,
        rf_runtime_,
    ) = acc_runtime(drugs_dict_, regression=regression_)

    # plot runtime and accuracies of each subset
    plot_runtime(
        rf_runtime_, "runtime", output_dir=os.path.join(output_dir, "runtimes")
    )
    plot_drugs_acc_subsets(
        regression_,
        best_model_acc_,
        all_params_acc_,
        params_acc_per_drug_,
        grid_params_,
        all_splits_acc_,
        output_dir,
        hspace=hspace,
    )

    # extract best model meta information
    best_models_info = best_est_info(drugs_dict_)

    # plot parameter selection of the best models
    plot_param_selection(
        best_models_info["best_params"],
        output_dir=os.path.join(output_dir, "parameters"),
    )

    # plot the feature importance of the first n feature for each model
    plot_features_importance(
        best_models_info["features_importance"],
        n_features,
        output_dir=os.path.join(output_dir, "feature_importance"),
    )
