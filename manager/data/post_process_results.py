import json
import math
import os
import re
import warnings
from ast import literal_eval
from math import inf
from statistics import mean

import pandas as pd


def serialize_dict(multi_level_dict, levels_to_index):
    """
    convert multi-key dictionary (json) to multilevel dataframe
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


def get_num_features(file):
    file_path = os.path.dirname(file)
    num_features_file = os.path.join(file_path, "num_features.json")
    num_features = None
    if os.path.isfile(num_features_file):
        num_features = []
        with open(num_features_file, "r", encoding="utf-8") as f:
            json_data = re.sub(r"}\s*{", "},{", f.read())
            num_features.extend(json.loads("[" + json_data + "]"))
        num_features = num_features[0]
    return num_features


def rename_models(
    drugs_dict, drug, models, prefix, suffix, file, models_to_compare=None
):
    """
    rename models with proper prefix and suffix according to the file name (train configuration)
    """

    num_features = get_num_features(file)
    for model, results in models.items():
        if models_to_compare is not None and model not in models_to_compare:
            continue

        if model == "parameters_grid":
            new_name = "parameters_grid"
        elif model == "original":
            if "sauron" in file:
                new_name = f"{prefix}rf_sauron"
            else:
                new_name = f"{prefix}rf"
        elif model == "subgraphilp":
            new_name = f"{prefix}subILP{suffix}"
        else:
            new_name = f"{prefix}{model}{suffix}"

        if num_features is not None and model != "parameters_grid":
            # invoked for the averaged results which mistakenly didn't include detailed num_features
            if model in num_features[drug]:
                results["parameters_combo_cv_results"]["0"]["stats"][
                    "num_features"
                ] = num_features[drug][model]["num_features"]
                results["parameters_combo_cv_results"]["0"]["stats"][
                    "num_features_overall"
                ] = num_features[drug][model]["num_features_overall"]
            else:
                if "random" in model:
                    curr_num_features = 0
                else:
                    features_lists = literal_eval(
                        results["parameters_combo_cv_results"]["0"][
                            "important_features"
                        ]
                    )
                    num_features_list = []
                    for features_list in features_lists:
                        df = pd.DataFrame(
                            features_list["data"],
                            columns=features_list["columns"],
                            index=features_list["index"],
                        )
                        num_features_list.append(len(df.index))
                    curr_num_features = sum(num_features_list) / len(features_lists)
                results["parameters_combo_cv_results"]["0"]["stats"][
                    "num_features"
                ] = curr_num_features
                results["parameters_combo_cv_results"]["0"]["stats"][
                    "num_features_overall"
                ] = curr_num_features

        if drug in drugs_dict:
            if new_name in drugs_dict[drug] and drugs_dict[drug][new_name] != results:
                warnings.warn(
                    f"{new_name} already exists with different values. Second instance is used. {file}"
                )
            drugs_dict[drug][new_name] = results
        else:
            drugs_dict[drug] = {new_name: results}


def aggregate_result_files(results_path, condition, targeted=False, all_features=False):
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
    drugs_dict = {}
    drugs_with_targets = None
    if targeted:
        targeted_folder = [
            folder for folder in results_folders if "targeted" in folder
        ][0]
        to_check = os.path.join(targeted_folder, "dummy_result.csv")
        if os.path.isfile(to_check):
            not_to_include = pd.read_csv(to_check, header=None).iloc[:, 0].to_list()

        # extract the drugs with targets from the first file (arbitrary choice, should be the same drugs set for all)
        target_file = [file for file in results_files if "targeted" in file]
        for tf in target_file:  # TODO: remove
            td = read_results_file(tf)
            print(tf, len(td.keys()))
        target_file = [file for file in results_files if "targeted" in file][0]
        target_dd = read_results_file(target_file)
        drugs_with_targets = list(target_dd.keys())

    models_to_compare = None
    if all_features:
        original_file = [file for file in results_files if "original" in file][0]
        original_dd = read_results_file(original_file)
        models_to_compare = original_dd[list(original_dd.keys())[0]].keys()

    for file in results_files:
        if "original" in file:
            if all_features:
                prefix0 = "all_"
            else:
                continue
        else:
            prefix0 = ""
        if "targeted" in file:
            if targeted:
                prefix1 = "targeted_"
            else:
                continue
        else:
            prefix1 = ""
        prefix = f"{prefix0}{prefix1}"

        if "biased" in file:
            suffix1 = "_bias"
        else:
            suffix1 = ""
        if "sauron" in file:
            suffix2 = "_sauron"
        else:
            suffix2 = ""
        suffix = f"{suffix1}{suffix2}"

        result_dict = read_results_file(file)
        print(file, len(result_dict.keys()))  # TODO: remove

        for drug, models in result_dict.items():
            if targeted and drug not in drugs_with_targets:
                # update drugs_dict with only drugs that have targets
                continue

            rename_models(
                drugs_dict,
                drug,
                models,
                prefix,
                suffix,
                file,
                models_to_compare=models_to_compare,
            )
    if len(drugs_dict) == 0:
        warnings.warn("no results found")
    else:
        if not_to_include is not None:
            for drug in not_to_include:
                drugs_dict.pop(drug)
    return drugs_dict


def add_best_model(
    all_models,
    models_performance,
    best_model,
    drug,
    best_by,
    best_models,
    drug_df,
    second_best,
):
    for model in all_models:
        if model in models_performance:
            if model == best_model:
                models_performance[model][drug] = best_by
            elif model in best_models:
                models_performance[model][drug] = abs(drug_df[model] - second_best)
            else:
                models_performance[model][drug] = -1
        else:
            if model == best_model:
                models_performance[model] = {drug: best_by}
            elif model in best_models:
                models_performance[model] = {drug: abs(drug_df[model] - second_best)}
            else:
                models_performance[model] = {drug: -1}


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
    all_models = None
    drug_won_by = {}
    models_count = {}
    for drug in test_scores.index.to_list():
        drug_df = test_scores.loc[drug].round(2)
        if all_models is None:
            all_models = drug_df.index.to_list()

        temp = drug_df.copy(deep=True).abs()
        best_by = None
        best_scores = set()
        second_best = 0
        for ctr in range(len(drug_df.unique())):
            if regression:
                best_score = temp.min()
                second_best = temp[temp.values != best_score].min()
                best_by = second_best - best_score
            else:
                best_score = temp.max()
                second_best = temp[temp.values != best_score].max()
                best_by = best_score - second_best

            if best_by >= thresh:
                best_scores.add(best_score)
                break
            elif math.isnan(second_best):
                if not len(best_scores) > 0:
                    best_scores = set(drug_df.values)
                second_best = min(best_scores)
                best_by = abs(max(best_scores) - min(best_scores))
                break
            else:
                best_scores.update([best_score, second_best])
                temp = temp[temp.values != second_best]
        best_models = drug_df[drug_df.isin(best_scores)].index.to_list()

        # if the random model is among the best models, halt analysis for current drug
        if any(["random" in model for model in best_models]):
            drug_won_by[drug] = ["by_random"]
            best_model = [model for model in best_models if "random" in model][0]
            if best_model in models_count:
                models_count[best_model] += 1
            else:
                models_count[best_model] = 1
            add_best_model(
                all_models,
                models_performance,
                best_model,
                drug,
                best_by,
                best_models,
                drug_df,
                second_best,
            )
            continue

        min_features = inf
        min_runtime = inf
        best_model = None
        won_by_series = []
        for model in best_models:
            if (
                len(
                    final_models_num_features[list(final_models_num_features.keys())[0]]
                )
                > 0
            ):
                model_features = final_models_num_features[drug][model]
            else:
                model_features = 0
            model_runtime = runtimes.loc[drug, model]
            if model_features < 0.95 * min_features:
                won_by_series.append([f"{best_model} --> {model}", "by_num_features"])
                min_features = model_features
                min_runtime = model_runtime
                best_model = model
            elif model_features == min_features and model_runtime < 0.95 * min_runtime:
                won_by_series.append([f"{best_model} --> {model}", "by_runtime"])
                min_features = model_features
                min_runtime = model_runtime
                best_model = model
        drug_won_by[drug] = won_by_series
        if best_model in models_count:
            models_count[best_model] += 1
        else:
            models_count[best_model] = 1
        add_best_model(
            all_models,
            models_performance,
            best_model,
            drug,
            best_by,
            best_models,
            drug_df,
            second_best,
        )
    for model in all_models:
        if model not in models_performance:
            models_performance[model] = {
                drug: -1 for drug in test_scores.index.to_list()
            }
        if model not in models_count:
            models_count[model] = 0
    models_count = pd.DataFrame(models_count, index=["count"]).transpose().squeeze()
    return models_performance, models_count, drug_won_by


def postprocessing(
    results_path,
    condition,
    targeted,
    regression,
    metrics,
    thresh,
    specific_models=None,
    all_features=False,
    params_acc_stat="mean",
):
    drugs_dict = aggregate_result_files(
        results_path, condition, targeted, all_features=all_features
    )

    runtimes = {}
    final_models_parameters = {}
    final_models_num_unique_features = {}
    final_models_avg_num_features = {}
    final_models_best_features = {}
    drug_info = {}

    final_models_acc = {}
    parameters_grid_acc = {}
    parameters_grid_acc_per_drug = {}
    cross_validation_splits_acc = {}
    for metric in metrics:
        final_models_acc[metric] = {}
        parameters_grid_acc[metric] = {}
        parameters_grid_acc_per_drug[metric] = {}

    parameters_grid = None
    for drug, rf_models in drugs_dict.items():
        runtimes[drug] = {}
        final_models_parameters[drug] = {}
        final_models_num_unique_features[drug] = {}
        final_models_avg_num_features[drug] = {}
        final_models_best_features[drug] = {}
        cross_validation_splits_acc[drug] = {}
        drug_info[drug] = {}
        for metric in metrics:
            final_models_acc[metric][drug] = {}
            parameters_grid_acc[metric][drug] = {}
            parameters_grid_acc_per_drug[metric][drug] = {}

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
                if specific_models is not None and rf_model not in specific_models:
                    continue
                drug_info[drug][rf_model] = {}

                # process file based on whether cv was used as average results or parameter tuning
                if "best_params_performance" in results:
                    best_model = results["best_params_performance"]
                else:
                    stats = results["parameters_combo_cv_results"]["0"]["stats"]

                    if regression:
                        suffix = "mean_mse"
                    else:
                        suffix = "mean"

                    imp = None
                    if "random" not in rf_model:
                        imp = stats["important_features"]

                    # "params" key should be fixed in analysis to one grid only
                    best_model = {
                        "params": rf_models["parameters_grid"]["0"],
                        "train_runtime": stats["train_time_mean"],
                        "test_runtime": stats["test_time_mean"],
                        "model_runtime": stats["train_time_mean"]
                        + stats["test_time_mean"],
                        "test_scores": {
                            metric: stats[f"{metric}_{suffix}"] for metric in metrics
                        },
                        "features_importance": imp,
                    }
                    if "num_features" in stats:
                        best_model["num_features_overall"] = stats[
                            "num_features_overall"
                        ]
                        best_model["num_features"] = stats["num_features"]
                    else:
                        best_model["num_features_overall"] = 0
                        best_model["num_features"] = 0

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
                    drug_info[drug][rf_model][metric] = best_model["test_scores"][
                        metric
                    ]
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

                final_models_num_unique_features[drug][rf_model] = best_model[
                    "num_features_overall"
                ]
                final_models_avg_num_features[drug][rf_model] = best_model[
                    "num_features"
                ]
                drug_info[drug][rf_model]["num_unique_features"] = best_model[
                    "num_features_overall"
                ]
                drug_info[drug][rf_model]["avg_num_features_per_tree"] = best_model[
                    "num_features"
                ]

                if "random" not in rf_model and not all_features:
                    dict_info = literal_eval(best_model["features_importance"])
                    final_models_best_features[drug][rf_model] = pd.DataFrame(
                        dict_info["data"],
                        columns=dict_info["columns"],
                        index=dict_info["index"],
                    )
                # endregion

                # region arrange cross validation splits accuracies
                parameters_scores = results["parameters_combo_cv_results"]
                cross_validation_splits_acc[drug][rf_model] = {}
                for metric in metrics:
                    if len(parameters_grid) == 1:
                        cross_validation_splits_acc[drug][rf_model][metric] = {
                            f"split_{cv_idx}": cv_acc
                            for cv_idx, cv_acc in enumerate(
                                list(parameters_scores.values())[0][f"ACCs_{metric}"]
                            )
                        }
                        continue

                    cross_validation_splits_acc[drug][rf_model][metric] = {}
                    for cv_results in parameters_scores.values():
                        metric_cv_acc = cv_results[f"ACCs_{metric}"]
                        for cv_idx, cv_acc in enumerate(metric_cv_acc):
                            if (
                                cv_idx
                                in cross_validation_splits_acc[drug][rf_model][metric]
                            ):
                                cross_validation_splits_acc[drug][rf_model][metric][
                                    f"split_{cv_idx}"
                                ].append(cv_acc)
                            else:
                                cross_validation_splits_acc[drug][metric][rf_model][
                                    f"split_{cv_idx}"
                                ] = [cv_acc]
                if len(parameters_grid) == 1:
                    cross_validation_splits_acc[drug][rf_model] = pd.DataFrame(
                        cross_validation_splits_acc[drug][rf_model]
                    )
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
    runtimes = serialize_dict(runtimes, [0]).swaplevel(0, 1, axis=1)
    final_models_parameters = serialize_dict(
        final_models_parameters, [2, 1]
    ).transpose()

    if all_features:
        all_features_acc = {}
        for metric, results in final_models_acc.items():
            all_features_acc[metric] = {}
            for drug, models in results.items():
                all_features_acc[metric][drug] = {}
                orig = [
                    model.replace("all_", "")
                    for model in list(models.keys())
                    if "all_" in model
                ]
                for model, scores in models.items():
                    if model in orig or "all" in model:
                        all_features_acc[metric][drug][model] = scores
    else:
        all_features_acc = None

    unique_num_features_df = pd.concat(
        [pd.DataFrame(final_models_num_unique_features).transpose()],
        axis=1,
        keys=["Number of Unique Features"],
    )
    avg_num_features_df = pd.concat(
        [
            pd.DataFrame(final_models_avg_num_features)
            .transpose()
            .loc[unique_num_features_df.index]
        ],
        axis=1,
        keys=["Average Number of Features per Tree"],
    )
    final_models_num_features = pd.concat(
        [unique_num_features_df, avg_num_features_df], axis=1
    )

    test_metric_best_models = {}
    test_metric_best_models_detailed = {}
    test_metric_best_models_count = {}
    cv_metric_best_models = {}
    cv_metric_best_models_detailed = {}
    cv_metric_best_models_count = {}
    for metric in metrics:
        final_models_acc[metric] = serialize_dict(
            final_models_acc[metric], 0
        ).swaplevel(axis=1)

        if all_features_acc is not None:
            all_features_acc[metric] = serialize_dict(
                all_features_acc[metric], 0
            ).swaplevel(axis=1)

        parameters_grid_acc[metric] = serialize_dict(
            parameters_grid_acc[metric], [2, 0]
        )

        parameters_grid_acc_per_drug[metric] = serialize_dict(
            parameters_grid_acc_per_drug[metric], [0]
        )

        (
            test_metric_best_models[metric],
            test_metric_best_models_count[metric],
            test_metric_best_models_detailed[metric],
        ) = best_model_per_drug(
            final_models_acc[metric]["test_score"],
            final_models_num_unique_features,
            runtimes["gcv_runtime"],
            regression,
            thresh,
        )
        (
            cv_metric_best_models[metric],
            cv_metric_best_models_count[metric],
            cv_metric_best_models_detailed[metric],
        ) = best_model_per_drug(
            final_models_acc[metric]["train_score"],
            final_models_num_unique_features,
            runtimes["gcv_runtime"],
            regression,
            thresh,
        )

    metric_best_models_count = {}
    metric_best_models = {}
    metric_best_models_detailed = {}
    for metric in test_metric_best_models_count:
        indices = test_metric_best_models_count[metric].index.to_list()
        metric_best_models_count[metric] = pd.DataFrame(
            [
                test_metric_best_models_count[metric].values,
                cv_metric_best_models_count[metric].loc[indices].values,
            ],
            columns=indices,
            index=["test_score", "train_score"],
        ).transpose()

        metric_best_models[metric] = {
            "test_score": test_metric_best_models[metric],
            "train_score": cv_metric_best_models[metric],
        }
        metric_best_models_detailed[metric] = {
            "test_score": test_metric_best_models_detailed[metric],
            "train_score": cv_metric_best_models_detailed[metric],
        }

    drugs_acc = {}
    for metric in final_models_acc.keys():
        metric_drugs = final_models_acc[metric]
        for drug in metric_drugs.index:
            if drug in drugs_acc:
                drugs_acc[drug][metric] = metric_drugs.loc[drug].unstack(level=0)
            else:
                drugs_acc[drug] = {metric: metric_drugs.loc[drug].unstack(level=0)}

    return (
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
    )


def postprocess_final(results_file):
    final_results = read_results_file(results_file)
    scores = {}
    for drug, results in final_results.items():
        for model, result in results.items():
            if model != "parameters_grid":
                for grid_idx, params_results in result.items():
                    if grid_idx in scores:
                        if drug in scores[grid_idx]:
                            if model in scores[grid_idx][drug]:
                                warnings.warn(
                                    f"{model} already exists for this parameter combination! "
                                    f"Second instance is ignored"
                                )
                                continue
                            scores[grid_idx][drug][model] = params_results[
                                "test_scores"
                            ]
                        else:
                            scores[grid_idx][drug] = {
                                model: params_results["test_scores"]
                            }
                    else:
                        scores[grid_idx] = {
                            drug: {model: params_results["test_scores"]}
                        }
    scores_df = {
        grid_idx: serialize_dict(idx_scores, [0]).swaplevel(0, 1, axis=1)
        for grid_idx, idx_scores in scores.items()
    }
    return scores_df


def process_trees_info(rf_trees_file):
    trees_json = []
    with open(rf_trees_file, "r", encoding="utf-8") as f:
        json_data = re.sub(r"}\s*{", "},{", f.read())
        trees_json.extend(json.loads("[" + json_data + "]"))

    features_importance = pd.DataFrame()
    for tree in trees_json:
        tree_index = list(tree.keys())[0]
        tree_info = tree[tree_index]
        tree_df = pd.DataFrame(
            {f"features_importance_{tree_index}": tree_info["features_importance"]},
            index=tree_info["features"],
        )
        features_importance = pd.concat([features_importance, tree_df], axis=1)

    num_trees_features = features_importance.count().to_list()
    features_stats = {
        "num_features_overall": len(features_importance.index),
        "num_trees_features": num_trees_features,
        "num_features": int(mean(num_trees_features)),
    }

    used_features_count = features_importance.count(axis=1)
    used_features_importance = features_importance.mean(axis=1)
    return used_features_count, used_features_importance, features_stats


def trees_summary(results_path, condition, all_features=False):
    all_folders = [
        os.path.join(results_path, folder)
        for folder in os.listdir(results_path)
        if condition in folder
    ]
    trees_folders = [
        os.path.join(folder, "rf_trees_info")
        for folder in all_folders
        if os.path.isdir(os.path.join(folder, "rf_trees_info"))
    ]
    trees_features_summary = {}
    trees_features_dist = {}
    for folder in trees_folders:
        for drug_name in os.listdir(folder):
            trees_features_summary[drug_name] = {}
            trees_features_dist[drug_name] = {}
            trees_folder = os.path.join(folder, drug_name)
            for model_trees in os.listdir(trees_folder):
                model = model_trees.split(".")[0]
                if model[-1].isnumeric():
                    model = model[:-2]
                    trees_files = [
                        os.path.join(trees_folder, m)
                        for m in os.listdir(trees_folder)
                        if m.startswith(model)
                    ]
                else:
                    trees_files = os.path.join(trees_folder, model_trees)

                if model == "random":
                    continue

                if not all_features and "original" in trees_files:
                    continue

                if isinstance(trees_files, list):
                    all_dist, all_importance = pd.DataFrame(), pd.DataFrame()
                    stats = {}
                    for tree_file in trees_files:
                        features_dist, importance, stat = process_trees_info(tree_file)
                        all_dist = pd.concat([all_dist, features_dist], axis=1)
                        all_importance = pd.concat([all_importance, importance], axis=1)
                        for k, v in stat.items():
                            if k in stats:
                                stats[k].append(v)
                            else:
                                stats[k] = [v]

                    features_dist = all_dist.mean(axis=1).astype(int)
                    importance = all_importance.mean(axis=1)
                    for k, v in stats.items():
                        if isinstance(v, list) and not isinstance(v[0], list):
                            stats[k] = int(mean(v))
                else:
                    features_dist, importance, stats = process_trees_info(trees_files)

                to_dict = {
                    "features_dist": features_dist.sort_values(ascending=False)
                    .reset_index()
                    .rename({"index": "GeneID", 0: "count"}, axis=1),
                    "features_importance": importance,
                    "stats": stats,
                }
                trees_features_summary[drug_name][model] = to_dict
                trees_features_dist[drug_name][model] = to_dict["features_dist"][
                    "count"
                ]
    models_features_imp = {}
    stats_dict = trees_features_summary.copy()
    for drug, models in trees_features_summary.items():
        models_features_imp[drug] = pd.DataFrame()
        for model, info in models.items():
            stats_dict[drug][model] = info["stats"]
            models_features_imp[drug] = pd.concat(
                [
                    models_features_imp[drug],
                    pd.DataFrame({model: info["features_importance"]}),
                ],
                axis=1,
            )

    with open(
        os.path.join(os.path.dirname(trees_folders[0]), "num_features.json"), "w"
    ) as stats_file:
        stats_file.write(json.dumps(stats_dict, indent=2))
    return trees_features_dist, trees_features_summary, models_features_imp
