import json
import os
import re
import warnings
from ast import literal_eval

import pandas as pd

classification_metrics = ["sensitivity", "specificity", "f1", "youden_j", "mcc"]
subsets = ["overall", "sensitivity", "specificity"]


def serialize_dict(multi_level_dict, levels_to_index):
    df = pd.json_normalize(multi_level_dict)
    df.columns = df.columns.str.split(".").map(tuple)
    df = df.stack(levels_to_index).reset_index(0, drop=True)
    return df


def get_files(results_path, condition):
    folders = [
        os.path.join(results_path, folder)
        for folder in os.listdir(results_path)
        if os.path.isdir(os.path.join(results_path, folder)) and condition in folder
    ]
    files = [
        os.path.join(folder, file)
        for folder in folders
        for file in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, file)) and condition in file
    ]
    return files


def read_results_file(input_file):
    drugs_dict = {}
    with open(input_file, "r", encoding="utf-8") as file:
        json_data = re.sub(r"}\s*{", "},{", file.read())  # when json file is indented
        for drug in json.loads("[" + json_data + "]"):
            for drug_name, results in drug.items():
                drugs_dict[drug_name] = results
    return drugs_dict


def get_targeted_drugs(drugs_dict):
    drugs = []
    for drug, results in drugs_dict.items():
        if results["subgraphilp"] is not None:
            drugs.append(drug)
    if len(drugs) > 0:
        return drugs
    else:
        warnings.warn("no results for drugs with targets found")
        return None


def fetch_models(drugs_dict, drug, models, prefix, suffix):
    for model, results in models.items():
        if model == "parameters_grid":
            new_name = "parameters_grid"
        elif model == "original":
            new_name = "rf"
        elif model == "subgraphilp":
            new_name = f"{prefix}{model}{suffix}"
        else:
            new_name = f"{model}{suffix}"
        if drug in drugs_dict:
            drugs_dict[drug][new_name] = results
        else:
            drugs_dict[drug] = {new_name: results}
    return drugs_dict


def aggregate_result_files(results_path, condition, targeted=False):
    files = get_files(results_path, condition)
    drugs_dict = {}
    drugs_with_targets = None
    if targeted:
        target_file = [file for file in files if "targeted" in file][
            0
        ]  # should be only one file
        target_dd = read_results_file(target_file)
        drugs_with_targets = get_targeted_drugs(target_dd)

    for file in files:
        if "sauron" in file:
            suffix1 = "_sauron"
        else:
            suffix1 = ""
        if "biased" in file:
            suffix2 = "rf_trees"
        else:
            suffix2 = "rf"
        suffix = f"{suffix1}_{suffix2}"
        if targeted and "targeted" in file:
            prefix = "targeted_"
        elif (
            not targeted and "targeted" in file
        ):  # not using targets' results for plotting
            continue
        else:
            prefix = ""

        dd = read_results_file(file)
        for drug, models in dd.items():
            if targeted and drugs_with_targets is not None:
                if drug in drugs_with_targets:
                    drugs_dict = fetch_models(drugs_dict, drug, models, prefix, suffix)
            elif not targeted:
                drugs_dict = fetch_models(drugs_dict, drug, models, prefix, suffix)
            else:
                warnings.warn("no results for drugs with targets found")
    return drugs_dict


def best_est_info(drugs_dict):
    params = {}
    features_imp = {}
    for drug, info in drugs_dict.items():
        params[drug] = {}
        features_imp[drug] = {}
        for rf_model, results in info.items():
            if (
                results is not None and rf_model != "parameters_grid"
            ):  # correlation_thresh may not report results
                best_est = results["best_params_performance"]
                if best_est is not None:  # correlation_thresh may not report results
                    params[drug][rf_model] = {
                        param: best_est["params"][param]
                        for param in ["max_features", "min_samples_leaf"]
                    }
                    params[drug][rf_model]["num_features"] = best_est["num_features"]
                    if "random" not in rf_model:
                        dict_info = literal_eval(best_est["features_importance"])
                        features_imp[drug][rf_model] = pd.DataFrame(
                            dict_info["data"],
                            columns=dict_info["columns"],
                            index=dict_info["index"],
                        )
    return {
        "best_params": serialize_dict(params, [2, 1]).transpose(),
        "features_importance": serialize_dict(features_imp, [0]),
    }


def ml_methods_best_est_info(drugs_summary):
    ml_est_info = {}
    for ml_method, drugs_dict in drugs_summary.items():
        ml_est_info[ml_method] = best_est_info(drugs_dict)
    return ml_est_info


def compare_params(estimators_dict, txt):
    df = pd.json_normalize(estimators_dict[txt.split("_")[1]]["best_params"])
    df.columns = df.columns.str.split(".").map(tuple)
    df = df.stack([-1, 1]).reset_index(0, drop=True).transpose()
    return df


def organize_results(
    drug,
    rf_model,
    regression,
    results,
    all_params_acc,
    params_acc_per_drug,
    all_splits_acc,
    best_model_acc,
    sub_metric_list,
    params_acc_stat="mean",
):

    best_est_fit_runtime = {}
    best_est_test_runtime = {}
    best_est_runtime = {}
    best_model_acc[drug][rf_model] = {}

    # region arrange cv accuracies
    for sub_metric in sub_metric_list:
        all_splits_acc[sub_metric][drug][rf_model] = {}
        for gs_idx, cv_results in results["parameters_combo_cv_results"].items():
            sub_metric_cv_acc = cv_results[f"ACCs_{sub_metric}"]
            for cv_idx, cv_acc in enumerate(sub_metric_cv_acc):
                if cv_idx in all_splits_acc[sub_metric][drug][rf_model]:
                    all_splits_acc[sub_metric][drug][rf_model][cv_idx].append(cv_acc)
                else:
                    all_splits_acc[sub_metric][drug][rf_model][cv_idx] = [cv_acc]
    # endregion

    # region arrange parameters accuracies
    cv_stats = {
        idx: cv["stats"] for idx, cv in results["parameters_combo_cv_results"].items()
    }
    cv_stats_df = pd.DataFrame(cv_stats).transpose()
    for sub_metric in sub_metric_list:
        if regression:
            all_params_acc[sub_metric][drug][rf_model] = {
                idx: val
                for idx, val in enumerate(
                    cv_stats_df[f"{sub_metric}_{params_acc_stat}_mse"].to_list()
                )
            }
            params_acc_per_drug[sub_metric][drug][rf_model] = cv_stats_df[
                f"{sub_metric}_{params_acc_stat}_mse"
            ].to_list()
        else:
            all_params_acc[sub_metric][drug][rf_model] = {
                idx: val
                for idx, val in enumerate(
                    cv_stats_df[f"{sub_metric}_{params_acc_stat}"].to_list()
                )
            }
            params_acc_per_drug[sub_metric][drug][rf_model] = cv_stats_df[
                f"{sub_metric}_{params_acc_stat}"
            ].to_list()
    # endregion

    # region best model accuracies
    ranked_scores = results["rank_params_scores"]
    best_params_idx = [
        list(rank.keys())[0]
        for sub_metric, rank in results["rank_params_scores"].items()
        if sub_metric == "sensitivity"
    ][0]
    if results["best_params_performance"] is not None:
        for sub_metric in sub_metric_list:
            best_model_acc[drug][rf_model][sub_metric] = {
                f"test_score": results["best_params_performance"]["test_scores"][
                    sub_metric
                ],
                f"train_score": ranked_scores[sub_metric][best_params_idx],
            }
    # endregion

    # region arrange runtimes
    gcv_runtime = results["gcv_runtime"]
    if results["best_params_performance"] is not None:
        best_est_fit_runtime[rf_model] = results["best_params_performance"][
            "train_runtime"
        ]
        best_est_test_runtime[rf_model] = results["best_params_performance"][
            "test_runtime"
        ]
        best_est_runtime[rf_model] = results["best_params_performance"]["model_runtime"]
    # endregion

    return (
        best_est_fit_runtime,
        best_est_test_runtime,
        best_est_runtime,
        gcv_runtime,
        best_model_acc,
        all_params_acc,
        params_acc_per_drug,
        all_splits_acc,
    )


def acc_runtime(drugs_dict, regression):
    best_model_acc = {}
    rf_runtime = {}

    if regression:
        sub_metric_list = subsets
    else:
        sub_metric_list = classification_metrics

    all_params_acc = {}
    params_acc_per_drug = {}
    all_splits_acc = {}
    for sub_metric in sub_metric_list:
        all_params_acc[sub_metric] = {}
        params_acc_per_drug[sub_metric] = {}
        all_splits_acc[sub_metric] = {}

    grid_params = None
    for drug, rf_models in drugs_dict.items():
        best_model_acc[drug] = {}
        rf_runtime[drug] = {}
        for sub_metric in sub_metric_list:
            all_params_acc[sub_metric][drug] = {}
            params_acc_per_drug[sub_metric][drug] = {}
            all_splits_acc[sub_metric][drug] = {}
        for rf_model, results in rf_models.items():
            if results is not None:
                # store parameters in dataframe format
                if rf_model == "parameters_grid":
                    if grid_params is None:
                        grid_params = pd.DataFrame(results).transpose()[
                            ["max_features", "min_samples_leaf"]
                        ]
                else:
                    rf_runtime[drug][rf_model] = {}
                    (
                        best_est_fit_runtime,
                        best_est_test_runtime,
                        best_est_runtime,
                        gcv_runtime,
                        best_model_acc,
                        all_params_acc,
                        params_acc_per_drug,
                        all_splits_acc,
                    ) = organize_results(
                        drug,
                        rf_model,
                        regression,
                        results,
                        all_params_acc,
                        params_acc_per_drug,
                        all_splits_acc,
                        best_model_acc,
                        sub_metric_list,
                    )
                    if rf_model in best_est_runtime:
                        rf_runtime[drug][rf_model][
                            "rf_train_runtime"
                        ] = best_est_fit_runtime[rf_model]
                        rf_runtime[drug][rf_model][
                            "rf_test_runtime"
                        ] = best_est_test_runtime[rf_model]
                        rf_runtime[drug][rf_model]["model_runtime"] = best_est_runtime[
                            rf_model
                        ]
                    rf_runtime[drug][rf_model]["gcv_runtime"] = gcv_runtime
    for sub_metric in sub_metric_list:
        all_params_acc[sub_metric] = serialize_dict(all_params_acc[sub_metric], [2, 0])
        params_acc_per_drug[sub_metric] = serialize_dict(
            params_acc_per_drug[sub_metric], [0]
        )

    grid_params.index = grid_params.index.astype(int)
    grid_params["min_samples_leaf"] = grid_params["min_samples_leaf"].astype(int)
    rf_runtime = serialize_dict(rf_runtime, [1, 0])
    return (
        grid_params,
        best_model_acc,
        all_params_acc,
        params_acc_per_drug,
        all_splits_acc,
        rf_runtime,
    )


def agg_df(df, key):
    aggregated_dict = {}
    grouped_df = df.groupby(level=0)
    aggregated_dict[f"{key}-mean"] = grouped_df.mean()
    aggregated_dict[f"{key}-median"] = grouped_df.median()
    aggregated_dict[f"{key}-std"] = grouped_df.std()
    return aggregated_dict


def aggregate_multi_level_dict(main_dict, regression, levels, rename=None):
    aggregated_dict = {}
    if regression:
        df_temp = serialize_dict(main_dict, levels)
        df_temp.rename(rename, inplace=True)
        for key in set([idx[0] for idx in df_temp.index]):
            df = df_temp.loc[key]
            key_aggregated_dict = agg_df(df, key)
            for k, v in key_aggregated_dict.items():
                aggregated_dict[k] = v
    for key, val in main_dict.items():
        df = serialize_dict(val, levels)
        df.rename(rename, inplace=True)
        key_aggregated_dict = agg_df(df, key)
        for k, v in key_aggregated_dict.items():
            aggregated_dict[k] = v
    return aggregated_dict


def ml_methods_acc_runtime(drugs_summary):
    acc_runtime_info = {}
    for ml_method, drugs_dict in drugs_summary.items():
        acc_runtime_info[ml_method] = {}
        if ml_method == "regression":
            regression = True
        else:
            regression = False
        best_model_acc, all_params_acc, rf_runtime = acc_runtime(drugs_dict, regression)
        acc_runtime_info[ml_method]["acc_dict"] = best_model_acc
        acc_runtime_info[ml_method]["runtime_to_plot"] = rf_runtime
        acc_runtime_info[ml_method]["all_params_avg"] = all_params_acc
    return acc_runtime_info
