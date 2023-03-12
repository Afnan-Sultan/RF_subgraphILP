import json
import os
import re
import warnings
from ast import literal_eval
from statistics import mean

import pandas as pd
from manager.utils import NewJsonEncoder
from manager.visualization.plot_results_analysis.plot_results import (
    plot_common_features_mat,
)
from tqdm import tqdm


def aggregate_result_files(
    results_path, condition, averaged_results=True, targeted=False, all_features=False
):
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
        if os.path.isfile(os.path.join(folder, file))
        and file.endswith("jsonl")
        and "features" not in file
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

    for file in results_files:
        if "original" in file and not all_features:
            continue
        if "targeted" in file and not targeted:
            continue

        result_dict = read_results_file(file, averaged_results)
        print(file, len(result_dict.keys()))  # TODO: remove

        prefix, suffix = models_names_fixes(file)
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


def read_results_file(input_file, averaged_results=True):
    """
    store the contest of the jsonl results file into a dictionary
    """
    drugs_dict = {}
    num_features = get_num_features(input_file)
    computed_num_features = {}
    with open(input_file, "r", encoding="utf-8") as file:
        json_data = re.sub(r"}\s*{", "},{", file.read())  # when json file is indented
        json_list = json.loads("[" + json_data + "]")

        output_dir = os.path.dirname(input_file)
        output_file = os.path.join(output_dir, "computed_num_features.json")
        for drug in json_list:
            for drug_name, results in drug.items():
                if drug_name not in computed_num_features:
                    computed_num_features[drug_name] = {}
                if averaged_results:
                    for model in results.keys():
                        if model != "parameters_grid":
                            computed_num_features[drug_name][model] = {}

                            # used for the models which wrongly didn't include num_features info
                            comp_num_features(
                                drug_name,
                                model,
                                results,
                                num_features,
                                computed_num_features,
                            )
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
        if not os.path.isfile(output_file):
            with open(output_file, "w") as out_json:
                out_json.write(json.dumps(computed_num_features, indent=2))
    return drugs_dict


def get_num_features(file):
    file_path = os.path.dirname(file)
    num_features_file1 = os.path.join(file_path, "num_features.json")
    num_features_file2 = os.path.join(file_path, "computed_num_features.json")
    num_features = None
    if os.path.isfile(num_features_file2):
        num_features = []
        with open(num_features_file2, "r", encoding="utf-8") as f:
            json_data = re.sub(r"}\s*{", "},{", f.read())
            num_features.extend(json.loads("[" + json_data + "]"))
        num_features = num_features[0]
    elif os.path.isfile(num_features_file1):
        num_features = []
        with open(num_features_file1, "r", encoding="utf-8") as f:
            json_data = re.sub(r"}\s*{", "},{", f.read())
            num_features.extend(json.loads("[" + json_data + "]"))
        num_features = num_features[0]
    return num_features


def comp_num_features(drug, model, results, num_features, computed_num_features):
    """
    invoked for the averaged results which mistakenly didn't include detailed num_features
    """
    stats = results[model]["parameters_combo_cv_results"]["0"]["stats"]
    cv_idx = results[model]["parameters_combo_cv_results"]["0"]
    if num_features is not None and model in num_features[drug]:
        stats["num_features"] = num_features[drug][model]["num_features"]
        stats["num_features_overall"] = num_features[drug][model][
            "num_features_overall"
        ]
        computed_num_features[drug][model]["num_features"] = num_features[drug][model][
            "num_features"
        ]
        computed_num_features[drug][model]["num_features_overall"] = num_features[drug][
            model
        ]["num_features_overall"]
    elif "num_features" not in stats:
        if "random" in model:
            curr_num_features = 0
        else:
            features_lists = [literal_eval(fl) for fl in cv_idx["important_features"]]
            num_features_list = []
            for features_list in features_lists:
                df = pd.DataFrame(
                    features_list["data"],
                    columns=features_list["columns"],
                    index=features_list["index"],
                )
                num_features_list.append(len(df.index))
            curr_num_features = int(sum(num_features_list) / len(features_lists))
        stats["num_features"] = curr_num_features
        stats["num_features_overall"] = curr_num_features
        computed_num_features[drug][model]["num_features"] = curr_num_features
        computed_num_features[drug][model]["num_features_overall"] = curr_num_features
    else:
        computed_num_features[drug][model]["num_features"] = stats["num_features"]
        computed_num_features[drug][model]["num_features_overall"] = stats[
            "num_features_overall"
        ]
    results[model]["parameters_combo_cv_results"]["0"]["stats"] = stats


def models_names_fixes(file):
    file_dir = os.path.basename(os.path.dirname(file))
    if "targeted" in file_dir:
        prefix = "targeted_"
    else:
        prefix = ""

    if "biased" in file_dir:
        suffix1 = "_bias"
    else:
        suffix1 = ""
    if "sauron" in file_dir:
        suffix2 = "_sauron"
    else:
        suffix2 = ""
    if "tuned" in file_dir:
        suffix3 = "_tuned"
    else:
        suffix3 = ""
    if "fc" in file_dir:
        suffix4 = "_fc"
    else:
        suffix4 = ""
    if "double" in file_dir:
        suffix5 = "_weighted_features"
    else:
        suffix5 = ""
    if "single_network" in file_dir:
        suffix6 = "_small"
    else:
        suffix6 = ""
    if "original" in file_dir:
        suffix7 = "_all_features"
    else:
        suffix7 = ""
    suffix = f"{suffix1}{suffix2}{suffix3}{suffix4}{suffix5}{suffix6}{suffix7}"
    return prefix, suffix


def rename_models(
    drugs_dict, drug, models, prefix, suffix, file, models_to_compare=None
):
    """
    rename models with proper prefix and suffix according to the file name (train configuration)
    """

    for model, results in models.items():
        if models_to_compare is not None and model not in models_to_compare:
            continue

        if model == "parameters_grid":
            new_name = "parameters_grid"
        elif model == "original":
            if "sauron" in file:
                new_name = f"{prefix}rf_sauron"
            elif "original" in file:
                new_name = f"rf{suffix}"
            else:
                new_name = "rf"
        elif model == "subgraphilp":
            new_name = f"{prefix}subILP{suffix}"
        else:
            new_name = f"{prefix}{model}{suffix}"

        if drug in drugs_dict:
            if new_name in drugs_dict[drug] and drugs_dict[drug][new_name] != results:
                warnings.warn(
                    f"{new_name} already exists with different values. Second instance is used. {file}"
                )
            drugs_dict[drug][new_name] = results
        else:
            drugs_dict[drug] = {new_name: results}


def add_best_model(
    models_count,
    all_models,
    models_performance,
    drug,
    best_models,
):
    for model in best_models:
        if model in models_count:
            models_count[model] += 1
        else:
            models_count[model] = 1

    for model in all_models:
        if model in models_performance:
            if model in best_models:
                models_performance[model][drug] = 1
            else:
                models_performance[model][drug] = 0
        else:
            if model in best_models:
                models_performance[model] = {drug: 1}
            else:
                models_performance[model] = {drug: 0}


def one_se_error(df, minimize=False):
    df_sem = df.sem()
    if minimize:
        best_score = df.min()
        acc_diff = df - best_score
    else:
        best_score = df.max()
        acc_diff = best_score - df
    best_models = acc_diff[acc_diff.values <= df_sem].index.to_list()
    return best_models


def best_model_per_drug(
    test_scores,
    final_models_num_features,
    runtimes,
    regression,
    use_num_features=False,
    use_runtime=False,
):
    """
    Determine the best performing model for each drug according to each metric (e.g., sensitivity/specificity).

    The one Standard Error (1se) rule is used to determine the models within on3 standard error form from the best
    performing model. A tie can be broke by repeating the 1se over the number of features by selecting the models
    within 1se from the minimum number of features. A further tie can be broken by runtime 1se.
    """
    models_performance = {}
    all_models = None
    models_count = {}
    filter_summary = pd.DataFrame()
    for drug in test_scores.index.to_list():
        to_csv = pd.DataFrame()
        drug_df = test_scores.loc[drug].round(2).abs()
        if all_models is None:
            # retrieve the tested models
            all_models = drug_df.index.to_list()

        if regression:
            best_models = one_se_error(drug_df, minimize=True)
        else:
            best_models = one_se_error(drug_df, minimize=False)
        to_csv = pd.concat([to_csv, drug_df.loc[best_models].rename("acc")], axis=1)

        # if the random model is among the best models, halt analysis for current drug
        if any(["random" in model for model in best_models]):
            best_models = [model for model in best_models if "random" in model]
            add_best_model(
                models_count,
                all_models,
                models_performance,
                drug,
                best_models,
            )
            continue

        # break ties by number of features and runtime
        if len(best_models) > 1:
            models_num_features = pd.Series(final_models_num_features[drug])
            best_num_features = models_num_features.loc[best_models]
            temp_best_models = one_se_error(best_num_features, minimize=True)
            to_csv = pd.concat(
                [
                    to_csv,
                    best_num_features.loc[temp_best_models].rename("num_features"),
                ],
                axis=1,
            )
            if use_num_features:
                best_models = temp_best_models

            if len(best_models) > 1:
                best_runtimes = runtimes.loc[drug, temp_best_models]
                temp_best_models = one_se_error(best_runtimes, minimize=True)
                to_csv = pd.concat(
                    [to_csv, best_runtimes.loc[temp_best_models].rename("runtime")],
                    axis=1,
                )
                if use_runtime:
                    best_models = temp_best_models
        if len(to_csv) > 0:
            filter_summary = pd.concat(
                [filter_summary, pd.concat([to_csv], keys=[drug])]
            )
        # increment the model count for the best models and store the current drug for these models
        add_best_model(
            models_count,
            all_models,
            models_performance,
            drug,
            best_models,
        )

    # assign count = 0 for the models that didn't do well on any drug
    for model in all_models:
        if model not in models_performance:
            models_performance[model] = {
                drug: 0 for drug in test_scores.index.to_list()
            }
        if model not in models_count:
            models_count[model] = 0
    models_count = pd.DataFrame(models_count, index=["count"]).transpose().squeeze()
    return models_performance, models_count, filter_summary


def get_best_model(results, regression, rf_models, rf_model, metrics):
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
            "model_runtime": stats["train_time_mean"] + stats["test_time_mean"],
            "test_scores": {metric: stats[f"{metric}_{suffix}"] for metric in metrics},
            "features_importance": imp,
        }
        if "num_features" in stats:
            best_model["num_features_overall"] = stats["num_features_overall"]
            best_model["num_features"] = stats["num_features"]
        else:
            best_model["num_features_overall"] = 0
            best_model["num_features"] = 0
    return best_model


def postprocessing(
    results_path,
    condition,
    targeted,
    regression,
    metrics,
    averaged_results=False,
    specific_models=None,
    all_features=False,
    params_acc_stat="mean",
    output_dir=None,
    csv_name=None,
):
    drugs_dict = aggregate_result_files(
        results_path=results_path,
        condition=condition,
        targeted=targeted,
        averaged_results=averaged_results,
        all_features=all_features,
    )

    runtimes = {}
    drug_info = {}
    final_models_parameters = {}
    final_models_num_unique_features = {}
    final_models_avg_num_features = {}
    final_models_best_features = {}
    cross_validation_splits_acc = {}

    final_models_acc = {}
    parameters_grid_acc = {}
    parameters_grid_acc_per_drug = {}
    for metric in metrics:
        final_models_acc[metric] = {}
        parameters_grid_acc[metric] = {}
        parameters_grid_acc_per_drug[metric] = {}

    parameters_grid = None
    for drug, rf_models in tqdm(drugs_dict.items(), desc="post-processing results..."):
        runtimes[drug] = {}
        drug_info[drug] = {}
        final_models_parameters[drug] = {}
        final_models_num_unique_features[drug] = {}
        final_models_avg_num_features[drug] = {}
        final_models_best_features[drug] = {}
        cross_validation_splits_acc[drug] = {}
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
                best_model = get_best_model(
                    results, regression, rf_models, rf_model, metrics
                )

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
                                cross_validation_splits_acc[drug][rf_model][metric][
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

    with open(os.path.join(output_dir, "tuned_params.json"), "w") as params_file:
        params_file.write(json.dumps(final_models_parameters))
    final_models_parameters = serialize_dict(
        final_models_parameters, [2, 1]
    ).transpose()

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
    test_metric_best_models_count = {}
    cv_metric_best_models = {}
    cv_metric_best_models_count = {}
    models_filter_summary = pd.DataFrame()
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

        (
            test_metric_best_models[metric],
            test_metric_best_models_count[metric],
            filter_summary,
        ) = best_model_per_drug(
            final_models_acc[metric]["test_score"],
            final_models_num_unique_features,
            runtimes["gcv_runtime"],
            regression,
        )
        if output_dir is not None:
            assert csv_name is not None
            models_filter_summary = pd.concat(
                [models_filter_summary, pd.concat([filter_summary], keys=[metric])]
            )
            models_filter_summary.to_csv(
                os.path.join(output_dir, f"{csv_name}_models_filter_summary.csv")
            )

        (
            cv_metric_best_models[metric],
            cv_metric_best_models_count[metric],
            _,
        ) = best_model_per_drug(
            final_models_acc[metric]["train_score"],
            final_models_num_unique_features,
            runtimes["gcv_runtime"],
            regression,
        )

    metric_best_models_count = {}
    metric_best_models = {}
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
    )


def recruit_drugs(metric_best_models):
    recruited_drugs = {}
    models_sens_performance = pd.DataFrame(
        metric_best_models["sensitivity"]["test_score"]
    )
    for row in models_sens_performance.iterrows():
        if row[1].sum() == 1:
            model = row[1][row[1] == 1].index.to_list()[0]
            if model in recruited_drugs:
                recruited_drugs[model].append(row[0])
            else:
                recruited_drugs[model] = [row[0]]
    return recruited_drugs


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


def serialize_dict(multi_level_dict, levels_to_index):
    """
    convert multi-key dictionary (json) to multilevel dataframe
    """
    df = pd.json_normalize(multi_level_dict)
    df.columns = df.columns.str.split(".").map(tuple)
    df = df.stack(levels_to_index).reset_index(0, drop=True)
    return df


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


def read_trees_summary(summary_file):
    trees_features_dist = {}
    with open(summary_file, "r", encoding="utf-8") as file:
        json_data = re.sub(r"}\s*{", "},{", file.read())
        data = json.loads(json_data)
        for drug, summary in data.items():
            trees_features_dist[drug] = {}
            for model, res in summary.items():
                dist_dict = literal_eval(res["features_dist"])
                data[drug][model]["features_dist"] = pd.DataFrame(
                    dist_dict["data"],
                    index=dist_dict["index"],
                    columns=dist_dict["columns"],
                )
                trees_features_dist[drug][model] = data[drug][model]["features_dist"][
                    "count"
                ]

                imp_dict = literal_eval(res["features_importance"].replace("null", "0"))
                data[drug][model]["features_importance"] = pd.Series(
                    imp_dict["data"], index=imp_dict["index"]
                )
    return data, trees_features_dist


def read_trees_imp_file(imp_file):
    with open(imp_file, "r", encoding="utf-8") as file:
        json_data = re.sub(r"}\s*{", "},{", file.read())
        data = json.loads(json_data)
        for drug, df_str in data.items():
            df_dict = literal_eval(df_str.replace("null", "111111"))
            data[drug] = pd.DataFrame(
                df_dict["data"], index=df_dict["index"], columns=df_dict["columns"]
            ).replace({111111: None})
    return data


def trees_summary(trees_folder, all_features=False):
    output_dir = os.path.dirname(trees_folder)
    summary_file = os.path.join(output_dir, "trees_features_summary.json")
    if os.path.isfile(summary_file):
        trees_features_summary, trees_features_dist = read_trees_summary(summary_file)
    else:
        trees_features_dist = {}
        trees_features_summary = {}
        print(trees_folder, "This might take quite long...")
        for drug_name in tqdm(
            os.listdir(trees_folder), desc="Processing trees per drug ..."
        ):
            trees_features_summary[drug_name] = {}
            trees_features_dist[drug_name] = {}
            trees_folder = os.path.join(trees_folder, drug_name)
            for model_trees in os.listdir(trees_folder):
                model = model_trees.split(".")[0]
                if model[-1].isnumeric():
                    model = f"{model[:-2]}_bias"
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
                        for k, count in stat.items():
                            if k in stats:
                                stats[k].append(count)
                            else:
                                stats[k] = [count]

                    features_dist = all_dist.mean(axis=1).astype(int)
                    importance = all_importance.mean(axis=1)
                    for k, count_list in stats.items():
                        if isinstance(count_list, list) and not isinstance(
                            count_list[0], list
                        ):
                            stats[k] = int(mean(count_list))
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
        with open(
            os.path.join(output_dir, "trees_features_dist.json"), "w"
        ) as dist_file:
            dist_file.write(
                json.dumps(trees_features_dist, indent=2, cls=NewJsonEncoder)
            )
        with open(
            os.path.join(output_dir, "trees_features_summary.json"), "w"
        ) as summary_file:
            summary_file.write(
                json.dumps(trees_features_summary, indent=2, cls=NewJsonEncoder)
            )

    imp_file = os.path.join(output_dir, "models_features_imp.json")
    if os.path.isfile(imp_file):
        models_features_imp = read_trees_imp_file(imp_file)
    else:
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
            os.path.join(output_dir, "models_features_imp.json"), "w"
        ) as imp_file:
            imp_file.write(
                json.dumps(models_features_imp, indent=2, cls=NewJsonEncoder)
            )
        with open(os.path.join(output_dir, "num_features.json"), "w") as stats_file:
            stats_file.write(json.dumps(stats_dict, indent=2, cls=NewJsonEncoder))
    return trees_features_dist, trees_features_summary, models_features_imp


def get_data_vs_prior_features_info(biased_file, ablation=False):
    trees_features_summary, _ = read_trees_summary(biased_file)
    features = {}
    num_features = {}
    for drug, models in trees_features_summary.items():
        features[drug] = {}
        num_features[drug] = {}
        for model, info in models.items():
            genes = info["features_dist"]["GeneID"].to_list()
            genes = [int(g) for g in genes]
            num_genes = info["stats"]["num_features_overall"]
            if model == "subgraphilp":
                features[drug]["subILP_bias"] = genes
                num_features[drug]["subILP_bias"] = num_genes
            else:
                if "bias" not in model:
                    model = f"{model}_bias"
                features[drug][model] = genes
                num_features[drug][model] = num_genes
    if ablation:
        return features, num_features
    else:
        num_features = pd.DataFrame(num_features).transpose()
        return features, num_features


def comp_features_intersection(
    models_features_importance,
    models_to_compare,
    n_features=None,
    selected=False,
    output_dir=None,
    name="",
):
    features_intersection = {}
    if isinstance(models_features_importance, list):
        self_intersection = False
        drugs_dict = models_features_importance[0]
    else:
        self_intersection = True
        drugs_dict = models_features_importance
    for drug, models in drugs_dict.items():
        if selected or not self_intersection:
            models_info = models
            features_intersection[drug] = {
                "common_features": {},
                "num_common": {},
            }
        else:
            models_info = {}
            for model in models_to_compare:
                temp = models[model].sort_values(ascending=False)[:n_features]
                if model == "subgraphilp":
                    model = "subILP"
                models_info[model] = temp

            features_intersection[drug] = {
                f"best_{n_features}_features": models_info,
                "common_features": {},
                "num_common": {},
            }
        for model, info in models_info.items():
            features_intersection[drug]["common_features"][model] = {}
            features_intersection[drug]["num_common"][model] = {}
            if self_intersection:
                inner_info = models_info
            else:
                inner_info = models_features_importance[1][drug]
            for model_again, info_again in inner_info.items():
                if selected or not self_intersection:
                    common_features = set(info).intersection(info_again)
                    num_common = len(common_features)
                else:
                    common_features = set(info.index).intersection(info_again.index)
                    num_common = len(common_features)

                if (
                    not selected
                    and self_intersection
                    and model == model_again
                    and num_common != n_features
                ):
                    # For some drugs, the selected num feature for corr_thresh can be very low that not enough
                    # {n_features} important features are present, which messes up the common_features_matrix.
                    # In this situation, the intersection between the model and itself is set to {n_features} to ensure
                    # logical matrix heatmap with unified diagonal colors.
                    num_common = n_features

                features_intersection[drug]["common_features"][model][
                    model_again
                ] = common_features
                features_intersection[drug]["num_common"][model][
                    model_again
                ] = num_common
        df = pd.DataFrame(features_intersection[drug]["num_common"]).loc[
            models_to_compare, models_to_compare
        ]
        if selected:
            features_intersection[drug]["num_common"] = (
                100 * (df / df.max(axis=1))
            ).round(0)
        else:
            features_intersection[drug]["num_common"] = df

    dfs = []
    for drug, info in features_intersection.items():
        dfs.append(info["num_common"])
    dfs_concat = pd.concat(dfs)
    dfs_concat.reset_index(inplace=True)
    mean_concat = dfs_concat.groupby("index").mean()
    features_intersection["Drugs' Average"] = {
        "num_common": mean_concat.loc[models_to_compare, models_to_compare]
    }

    if output_dir is not None:
        with open(
            os.path.join(output_dir, f"{name}best_features.json"), "w"
        ) as features_file:
            features_file.write(
                json.dumps(features_intersection, indent=2, cls=NewJsonEncoder)
            )
    return features_intersection


def get_features_and_importance(
    results_path,
    to_compare,
    regression,
    metrics,
    condition,
    targeted,
    averaged_results,
    all_features,
    output_dir,
    name="",
    ablation=True,
    corr_file=None,
    subilp_file=None,
    biased_file=None,
):
    drugs_dict = aggregate_result_files(
        results_path=results_path,
        condition=condition,
        targeted=targeted,
        averaged_results=averaged_results,
        all_features=all_features,
    )
    most_important_features = {}
    for drug, rf_models in drugs_dict.items():
        most_important_features[drug] = {}
        for rf_model, results in rf_models.items():
            if rf_model in to_compare:
                best_model = get_best_model(
                    results, regression, rf_models, rf_model, metrics
                )
                dict_info = literal_eval(best_model["features_importance"])
                temp_df = pd.DataFrame(
                    dict_info["data"],
                    columns=dict_info["columns"],
                    index=dict_info["index"],
                )
                most_important_features[drug][rf_model] = pd.Series(
                    temp_df["feature_importance"].values, index=temp_df["GeneSymbol"]
                )
    with open(
        os.path.join(output_dir, f"{name}important_features.json"), "w"
    ) as features_file:
        features_file.write(
            json.dumps(most_important_features, indent=2, cls=NewJsonEncoder)
        )
    if ablation:
        features, num_features = get_bias_ablation_features_info(
            corr_file, subilp_file, biased_file
        )
        num_features = pd.DataFrame(num_features).transpose()
        return features, num_features, most_important_features
    else:
        return most_important_features


def get_bias_ablation_features_info(corr_file, subilp_file, biased_file):
    with open(subilp_file, "r", encoding="utf-8") as file:
        json_data = re.sub(r"}\s*{", "},{", file.read())
        subilp_info = json.loads(json_data)
    with open(corr_file, "r", encoding="utf-8") as file:
        json_data = re.sub(r"}\s*{", "},{", file.read())
        json_list = json.loads("[" + json_data + "]")
        corr_info = {}
        for entry in json_list:
            for drug, model in entry.items():
                if drug not in corr_info:
                    corr_info[drug] = {}
                for model_name, info in model.items():
                    corr_info[drug][model_name] = info
    features, num_features = get_data_vs_prior_features_info(biased_file, ablation=True)
    for drug, models in corr_info.items():
        for model, info in models.items():
            features[drug][model] = corr_info[drug][model]["features"]
            num_features[drug][model] = corr_info[drug][model]["num_features"]

            features[drug]["subILP"] = subilp_info[drug]["features"]
            num_features[drug]["subILP"] = subilp_info[drug]["num_features"]
    return features, num_features


def read_best_imp_file(imp_file):
    with open(imp_file, "r", encoding="utf-8") as file:
        json_data = re.sub(r"}\s*{", "},{", file.read())
        data = json.loads(json_data)
        imp_dict = {}
        for drug, info_dict in data.items():
            imp_dict[drug] = {}
            if drug == "Drugs' Average":
                continue
            for model, series_str in info_dict["best_100_features"].items():
                df_dict = literal_eval(series_str.replace("null", "111111"))
                imp_dict[drug][model] = df_dict["index"]
    return imp_dict


def common_imp_feat_regression_n_classification(regression_file, classification_file):
    regression_features = read_best_imp_file(regression_file)
    classification_features = read_best_imp_file(classification_file)
    models = list(regression_features[list(regression_features.keys())[0]].keys())
    important_features_intersection = comp_features_intersection(
        [regression_features, classification_features], models_to_compare=models
    )
    return important_features_intersection


if __name__ == "__main__":
    ablation = False
    if ablation:
        classification_file = "../../figures_v4/weighted/classification/not_targeted/bias_ablation_best_features.json"
        regression_file = "../../figures_v4/weighted/regression/not_targeted/bias_ablation_best_features.json"
        name = "Regression and Classification - Bias Ablation"
    else:
        classification_file = "../../figures_v4/weighted/classification/not_targeted/data_vs_prior_best_features.json"
        regression_file = "../../figures_v4/weighted/regression/not_targeted/data_vs_prior_best_features.json"
        name = "Regression and Classification - Prior knowledge VS statistical"

    imp_features_intersections = common_imp_feat_regression_n_classification(
        regression_file, classification_file
    )
    df = imp_features_intersections["Drugs' Average"]["num_common"]
    drug = "Drugs' Average"
    plot_dir = "../../figures_v4"
    plot_common_features_mat(
        df, drug, plot_dir, figsize=(10, 10), name=name, title_pos=0.95, limits=(0, 100)
    )
