import json
import os
import re
import warnings
from ast import literal_eval
from math import inf

import pandas as pd


def serialize_dict(multi_level_dict, levels_to_index):
    """
    convert multi-key dictionary to multilevel dataframe
    """
    df = pd.json_normalize(multi_level_dict)
    df.columns = df.columns.str.split(".").map(tuple)
    df = df.stack(levels_to_index).reset_index(0, drop=True)
    return df


def read_results_file(input_file):
    """
    store the contest of the jsonl results file into a dictionary
    """
    drugs_dict = {}
    with open(input_file, "r", encoding="utf-8") as file:
        json_data = re.sub(r"}\s*{", "},{", file.read())  # when json file is indented
        for drug in json.loads("[" + json_data + "]"):
            for drug_name, results in drug.items():
                if drug_name in drugs_dict:
                    # If the same configuration is used with different models, the results is appended to the same
                    # results file. Hence, repeated drugs with different results can be present.
                    for model, scores in results.items():
                        if model not in drugs_dict[drug_name]:
                            drugs_dict[drug_name][model] = scores
                        elif (
                            model in drugs_dict[drug_name]
                            and model == "parameters_grid"
                        ):
                            if drugs_dict[drug_name][model] != scores:
                                warnings.warn(
                                    f"Parameters grid is repeated with different configuration for {drug_name}. "
                                    f"This might lead to faulty analysis"
                                )
                        else:
                            warnings.warn(
                                f"{model} for {drug_name} is already present. The second occurrence is skipped"
                            )
                else:
                    drugs_dict[drug_name] = results
    return drugs_dict


def fetch_models(drugs_dict, drug, models, prefix, suffix, file):
    """
    rename models with proper prefix and suffix according to the file name (train configuration)
    """
    for model, results in models.items():
        if model == "parameters_grid":
            new_name = "parameters_grid"
        elif model == "original":
            if "sauron" in file:
                new_name = "rf_sauron"
            else:
                new_name = "rf"
        elif model == "subgraphilp":
            new_name = f"{prefix}subILP{suffix}"
        else:
            new_name = f"{model}{suffix}"

        if drug in drugs_dict:
            if new_name in drugs_dict[drug] and drugs_dict[drug][new_name] != results:
                warnings.warn(
                    f"{new_name} already exists with different values. Second instance is used."
                )
            drugs_dict[drug][new_name] = results
        else:
            drugs_dict[drug] = {new_name: results}


def aggregate_result_files(results_path, condition, targeted=False):
    """
    different configurations are run in parallel for speed. this function aggregates all results in one dictionary
    """
    results_folders = [
        os.path.join(results_path, folder)
        for folder in os.listdir(results_path)
        if os.path.isdir(os.path.join(results_path, folder)) and condition in folder
    ]
    results_files = [
        os.path.join(folder, file)
        for folder in results_folders
        for file in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, file)) and file.endswith("jsonl")
    ]

    not_to_include = None
    if targeted:
        targeted_folder = [
            folder for folder in results_folders if "targeted" in folder
        ][0]
        to_check = os.path.join(targeted_folder, "dummy_result.csv")
        if os.path.isfile(to_check):
            not_to_include = pd.read_csv(to_check, header=None).iloc[:, 0].to_list()

    drugs_dict = {}
    drugs_with_targets = None
    if targeted:
        # extract the drugs with targets from the first file (arbitrary choice, should be the same drugs set for all)
        target_file = [file for file in results_files if "targeted" in file]
        for tf in target_file:  # TODO: remove
            td = read_results_file(tf)
            print(tf, len(td.keys()))
        target_file = [file for file in results_files if "targeted" in file][0]
        target_dd = read_results_file(target_file)
        drugs_with_targets = list(target_dd.keys())

    for file in results_files:
        if "biased" in file:
            suffix1 = "_bias"
        else:
            suffix1 = ""
        if "sauron" in file:
            suffix2 = "_sauron"
        else:
            suffix2 = ""
        suffix = f"{suffix1}{suffix2}"
        if targeted and "targeted" in file:
            prefix = "targeted_"
        elif not targeted and "targeted" in file:
            # not using targets' results for plotting
            continue
        else:
            prefix = ""

        result_dict = read_results_file(file)
        print(file, len(result_dict.keys()))  # TODO: remove
        for drug, models in result_dict.items():
            if not targeted:
                # update drugs_dict with all drugs
                fetch_models(drugs_dict, drug, models, prefix, suffix, file)
            elif targeted and drug in drugs_with_targets:
                # update drugs_dict with only drugs that have targets
                fetch_models(drugs_dict, drug, models, prefix, suffix, file)
    if len(drugs_dict) == 0:
        warnings.warn("no results found")
    else:
        if not_to_include is not None:
            for drug in not_to_include:
                drugs_dict.pop(drug)
    return drugs_dict


def best_model_per_drug(
    test_scores, final_models_num_features, runtimes, regression, thresh
):
    """
    Determine the best performing model for each drug according to each metric (e.g., sensitivity/specificity).

    The scores are rounded to second decimal to avoid wining over miniscule difference. Also, a model is considered
    amongst best models only if the difference in performance is higher than a threshold compared with the second model.
    If this difference didn't exist, the second-best models are considered best models as well and the comparison keeps
    moving on until the difference is realized.

    A tie is broken to favor the simpler model represented by lower number of features, then by lower runtime.
    In case the random model was one of the best models, all other models are ignored with the assumption that a random
    model is the simplest model possible.
    """
    models_performance = {}
    for drug in test_scores.index.to_list():
        drug_df = test_scores.loc[drug].round(2)
        temp = drug_df.copy(deep=True)
        best_by = None
        if regression:
            best_scores = set()
            for ctr in range(len(drug_df.unique())):
                best_score = temp.min()

                second_best = temp[temp.values != best_score].min()
                best_by = second_best - best_score
                if best_by >= thresh:
                    best_scores.add(best_score)
                    break
                else:
                    best_scores.update([best_score, second_best])
                    temp = temp[temp.values != second_best]
        else:
            best_scores = set()
            for ctr in range(len(drug_df.unique())):
                best_score = drug_df.max()

                second_best = drug_df[drug_df.values != best_score].max()
                best_by = best_score - second_best
                if best_by >= thresh:
                    best_scores.add(best_score)
                    break
                else:
                    best_scores.update([best_score, second_best])
                    temp = temp[temp.values != second_best]

        best_models = drug_df[drug_df.isin(best_scores)].index.to_list()
        if any(["random" in model for model in best_models]):
            best_model = [model for model in best_models if "random" in model][0]
            if best_model in models_performance:
                models_performance[best_model].append((drug, best_by))
            else:
                models_performance[best_model] = [(drug, best_by)]
            continue

        min_features = inf
        min_runtime = inf
        best_model = None
        for model in best_models:
            model_features = final_models_num_features[drug][model]
            model_runtime = runtimes.loc[model].loc[drug]
            if model_features is None and "random" in model:
                to_replace = "_".join(["corr_num", *model.split("_")[1:]])
                model_features = (
                    final_models_num_features[drug][to_replace]
                    if to_replace in final_models_num_features[drug]
                    else inf
                )
            if model_features < min_features or (
                model_features == min_features and model_runtime < min_runtime
            ):
                min_features = model_features
                min_runtime = model_runtime
                best_model = model
        if best_model in models_performance:
            models_performance[best_model].append((drug, best_by))
        else:
            models_performance[best_model] = [(drug, best_by)]
    return models_performance


def postprocessing(
    results_path,
    condition,
    targeted,
    regression,
    metrics,
    thresh,
    params_acc_stat="mean",
):
    drugs_dict = aggregate_result_files(results_path, condition, targeted)

    runtimes = {}
    final_models_parameters = {}
    final_models_num_features = {}
    final_models_best_features = {}

    final_models_acc = {}
    parameters_grid_acc = {}
    parameters_grid_acc_per_drug = {}
    cross_validation_splits_acc = {}
    for metric in metrics:
        final_models_acc[metric] = {}
        parameters_grid_acc[metric] = {}
        parameters_grid_acc_per_drug[metric] = {}
        cross_validation_splits_acc[metric] = {}

    parameters_grid = None
    for drug, rf_models in drugs_dict.items():
        runtimes[drug] = {}
        final_models_parameters[drug] = {}
        final_models_num_features[drug] = {}
        final_models_best_features[drug] = {}
        for metric in metrics:
            final_models_acc[metric][drug] = {}
            parameters_grid_acc[metric][drug] = {}
            parameters_grid_acc_per_drug[metric][drug] = {}
            cross_validation_splits_acc[metric][drug] = {}

        for rf_model, results in rf_models.items():
            if rf_model == "parameters_grid":
                if parameters_grid is None:
                    # store only once as it's assumed to be the same for all drugs and models
                    parameters_grid = pd.DataFrame(results).transpose()[
                        ["max_features", "min_samples_leaf"]
                    ]
                    parameters_grid.index = parameters_grid.index.astype(int)
                    parameters_grid["min_samples_leaf"] = parameters_grid[
                        "min_samples_leaf"
                    ].astype(int)
            else:
                best_model = results["best_params_performance"]

                # region final model accuracies
                ranked_scores = results["rank_params_scores"]
                final_models_parameters_idx = [
                    list(rank.keys())[0]
                    for metric, rank in results["rank_params_scores"].items()
                    if metric == "sensitivity"
                ][0]
                for metric in metrics:
                    final_models_acc[metric][drug][rf_model] = {
                        f"test_score": best_model["test_scores"][metric],
                        f"train_score": ranked_scores[metric][
                            final_models_parameters_idx
                        ],
                    }
                # endregion

                # region arrange final model's runtimes
                runtimes[drug][rf_model] = {
                    "gcv_runtime": results["gcv_runtime"],
                    "rf_train_runtime": best_model["train_runtime"],
                    "rf_test_runtime": best_model["test_runtime"],
                    "model_runtime": best_model["model_runtime"],
                }
                # endregion

                # region final model features and parameters
                final_models_parameters[drug][rf_model] = {
                    param: best_model["params"][param]
                    for param in ["max_features", "min_samples_leaf"]
                }

                final_models_num_features[drug][rf_model] = best_model[
                    "num_features_overall"
                ]

                if "random" not in rf_model:
                    dict_info = literal_eval(best_model["features_importance"])
                    final_models_best_features[drug][rf_model] = pd.DataFrame(
                        dict_info["data"],
                        columns=dict_info["columns"],
                        index=dict_info["index"],
                    )
                # endregion

                # region arrange cross validation splits accuracies
                parameters_scores = results["parameters_combo_cv_results"]
                for metric in metrics:
                    if len(parameters_grid) < 2:
                        cross_validation_splits_acc[metric][drug][rf_model] = {
                            f"split_{cv_idx}": cv_acc
                            for cv_idx, cv_acc in enumerate(
                                list(parameters_scores.values())[0][f"ACCs_{metric}"]
                            )
                        }
                        continue

                    cross_validation_splits_acc[metric][drug][rf_model] = {}
                    for cv_results in parameters_scores.values():
                        metric_cv_acc = cv_results[f"ACCs_{metric}"]
                        for cv_idx, cv_acc in enumerate(metric_cv_acc):
                            if (
                                cv_idx
                                in cross_validation_splits_acc[metric][drug][rf_model]
                            ):
                                cross_validation_splits_acc[metric][drug][rf_model][
                                    f"split_{cv_idx}"
                                ].append(cv_acc)
                            else:
                                cross_validation_splits_acc[metric][drug][rf_model][
                                    f"split_{cv_idx}"
                                ] = [cv_acc]
                # endregion

                # region arrange hyperparameter tuning accuracies
                cv_stats = {
                    idx: cv["stats"]
                    for idx, cv in results["parameters_combo_cv_results"].items()
                }
                cv_stats_df = pd.DataFrame(cv_stats).transpose()
                for metric in metrics:
                    if regression:
                        parameters_grid_acc[metric][drug][rf_model] = {
                            idx: val
                            for idx, val in enumerate(
                                cv_stats_df[f"{metric}_{params_acc_stat}_mse"].to_list()
                            )
                        }
                        parameters_grid_acc_per_drug[metric][drug][
                            rf_model
                        ] = cv_stats_df[f"{metric}_{params_acc_stat}_mse"].to_list()
                    else:
                        parameters_grid_acc[metric][drug][rf_model] = {
                            idx: val
                            for idx, val in enumerate(
                                cv_stats_df[f"{metric}_{params_acc_stat}"].to_list()
                            )
                        }
                        parameters_grid_acc_per_drug[metric][drug][
                            rf_model
                        ] = cv_stats_df[f"{metric}_{params_acc_stat}"].to_list()
                # endregion
    runtimes = serialize_dict(runtimes, [1, 0])
    final_models_parameters = serialize_dict(
        final_models_parameters, [2, 1]
    ).transpose()

    metric_best_models = {}
    for metric in metrics:
        final_models_acc[metric] = serialize_dict(
            final_models_acc[metric], 0
        ).swaplevel(axis=1)
        parameters_grid_acc[metric] = serialize_dict(
            parameters_grid_acc[metric], [2, 0]
        )
        parameters_grid_acc_per_drug[metric] = serialize_dict(
            parameters_grid_acc_per_drug[metric], [0]
        )

        metric_best_models[metric] = best_model_per_drug(
            final_models_acc[metric]["test_score"],
            final_models_num_features,
            runtimes["gcv_runtime"],
            regression,
            thresh,
        )

    if len(parameters_grid) < 2:
        for metric in metrics:
            for drug in cross_validation_splits_acc[metric].keys():
                cross_validation_splits_acc[metric][drug] = pd.DataFrame(
                    cross_validation_splits_acc[metric][drug]
                ).transpose()

    return (
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
    )
