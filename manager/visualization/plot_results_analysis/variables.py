import os

col_map = {
    "subILP": "thistle",
    "subILP_bias": "plum",
    "subILP_sauron": "violet",
    "subILP_bias_sauron": "purple",
    "subILP_bias_tuned": "pink",
    "subILP_bias_fc": "pink",
    "subILP_bias_small": "pink",
    "subILP_bias_weighted_features": "pink",
    "targeted_subILP": "lightsteelblue",
    "targeted_subILP_bias": "cornflowerblue",
    "targeted_subILP_sauron": "royalblue",
    "targeted_subILP_bias_sauron": "navy",
    "rf": "peachpuff",
    "rf_sauron": "sandybrown",
    "rf_all_features": "indianred",
    "random": "lightgrey",
    "random_bias": "grey",
    "corr_num": "lemonchiffon",
    "corr_num_bias": "khaki",
    "corr_num_sauron": "darkkhaki",
    "corr_num_bias_sauron": "olive",
    "corr_thresh": "paleturquoise",
    "corr_thresh_bias": "mediumturquoise",
    "corr_thresh_sauron": "lightseagreen",
    "corr_thresh_bias_sauron": "teal",
    "corr_thresh_bias_tuned": "lightblue",
    "corr_thresh_bias_weighted_features": "lightblue",
    "corr_thresh_all_features": "lightblue",
}


def output_mkdir(weighted, simple_weight, regression, targeted, output_path):
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

    output_dir = os.path.join(output_path, dir0, dir1, dir2)
    per_drug_dir = os.path.join(output_dir, "per_drug")
    os.makedirs(output_dir, exist_ok=True)
    condition = f"{dir1}_{dir0}"
    return condition, output_dir, per_drug_dir


def get_metrics(regression=True):
    if regression:
        metrics = {
            "overall": "$MSE$",
            "sensitivity": "$MSE_{sens}$",
            "specificity": "$MSE_{res}$",
        }
        title = f"Regression Performance"
        arranged_metrics = ["sensitivity", "specificity", "overall"]
    else:
        metrics = {
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "f1": "F1-score",
            "youden_j": "Youden's J",
            "mcc": "MCC",  # "Matthews Correlation Coefficient",
        }
        arranged_metrics = ["sensitivity", "specificity", "mcc"]
        title = "Classification Performance"
    return metrics, title, arranged_metrics


def analysis_utils(
    analysis_list,
    regression,
    targeted,
    condition,
    title,
    averaged_results=False,
    accuracies=False,
    fc=False,
    double_weighted=False,
    less_features=False,
    original=False,
):
    sub_analysis = []
    if regression:
        row_title_xpos = 0.45
    else:
        row_title_xpos = 0.3
    fig_height = 12
    for analysis_type in analysis_list:
        params = {
            "title": title,
            "row_title_xpos": row_title_xpos,
            "fig_height": fig_height,
        }
        if analysis_type == "rf_vs_subilp":
            assert not targeted
            params["fig_name"] = f"rf_vs_subilp_{condition}"
            params["fig_width"] = 15.0
            if not regression:
                # params["row_title_xpos"] = 0.5
                params["fig_width"] = 18.0
            params["specific_models"] = [
                "rf",
                "subILP_bias",
                "random_bias",
            ]
        elif analysis_type == "with_sauron":
            assert regression
            params["title"] = f"{title} - SAURON"
            params["fig_name"] = f"with_sauron_{condition}"
            params["fig_width"] = 17.0
            params["specific_models"] = [
                "rf",
                "subILP_bias",
                "random_bias",
                "rf_sauron",
                "subILP_bias_sauron",
            ]
        elif analysis_type == "targeted":
            params["title"] = f"{title} - Targeted subgraphILP"
            params["fig_name"] = f"targeted_{condition}"
            if regression:
                params["fig_width"] = 23.0
                params["specific_models"] = [
                    "rf",
                    "subILP_bias",
                    "random_bias",
                    "rf_sauron",
                    "subILP_bias_sauron",
                    "targeted_subILP_bias_sauron",
                    "targeted_subILP_bias",
                ]
            else:
                params["row_title_xpos"] = 0.5
                params["fig_width"] = 18.0
                params["specific_models"] = [
                    "rf",
                    "subILP_bias",
                    "random_bias",
                    "targeted_subILP_bias",
                ]
        elif analysis_type == "data_vs_prior":
            params["title"] = f"{title} - Methods Comparison"
            params["fig_name"] = f"data_vs_prior_{condition}"
            if targeted:
                params["title"] = f"Targeted {title} - Methods Comparison"
                params["fig_name"] = f"targeted_data_vs_prior_{condition}"
            if regression:
                params["fig_width"] = 20.0
                params["specific_models"] = [
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
                    params["fig_width"] = 22.5
                    params["specific_models"].extend(
                        [
                            "targeted_subILP_bias_sauron",
                            "targeted_subILP_bias",
                        ]
                    )
            else:
                params["fig_width"] = 18.0
                params["specific_models"] = [
                    "rf",
                    "subILP_bias",
                    "random_bias",
                    "corr_num_bias",
                    "corr_thresh_bias",
                ]
                if targeted:
                    params["fig_width"] = 18.0
                    params["specific_models"].append("targeted_subILP_bias")
        elif analysis_type == "num_features":
            params["fig_name"] = f"num_features"
            params["fig_height"] = 7
            params["fig_width"] = 13.0
            params["specific_models"] = [
                "subILP_bias",
                "corr_num_bias",
                "corr_thresh_bias",
            ]
            if targeted:
                params["specific_models"].append("targeted_subILP_bias")
        elif analysis_type == "without_bias":
            params["title"] = f"{title} - Bias Ablation"
            params["fig_name"] = f"without_bias_{condition}"
            if targeted:
                params["title"] = f"{title} - Targeted subgraphILP - Bias Ablation"
                params["fig_name"] = f"targeted_without_bias_{condition}"
            if regression:
                params["fig_width"] = 20.0
                params["specific_models"] = [
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
                    params["fig_width"] = 23.0
                    params["specific_models"].extend(
                        [
                            "targeted_subILP",
                            "targeted_subILP_bias",
                            "targeted_subILP_bias_sauron",
                        ]
                    )
            else:
                params["fig_width"] = 18.0
                params["specific_models"] = [
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
                    params["fig_width"] = 18.0
                    params["specific_models"].extend(
                        ["targeted_subILP", "targeted_subILP_bias"]
                    )
        elif analysis_type == "without_synergy":
            assert regression
            params["title"] = f"{title} - Synergy Ablation"
            params["fig_name"] = f"without_synergy_{condition}"
            if targeted:
                params["title"] = f"{title} - Targeted subgraphILP - Synergy Ablation"
                params["fig_name"] = f"targeted_without_synergy_{condition}"
            params["fig_width"] = 20.0
            params["specific_models"] = [
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
                params["fig_width"] = 23.0
                params["specific_models"].extend(
                    [
                        "targeted_subILP_bias_sauron",
                        "targeted_subILP_sauron",
                        "targeted_subILP_bias",
                    ]
                )
        elif analysis_type == "all":
            params["fig_name"] = f"performance_{condition}"
            params["fig_width"] = 21.0
            if not averaged_results:
                if accuracies:
                    params["specific_models"] = [
                        "corr_thresh_bias",
                        "corr_thresh_bias_tuned",
                        "subILP_bias",
                        "subILP_bias_tuned",
                    ]
                else:
                    params["specific_models"] = [
                        "corr_thresh_bias",
                        "subILP_bias",  # _tuned",  # _tuned",
                    ]
            elif fc:
                params["fig_width"] = 15.0
                params["specific_models"] = [
                    "subILP_bias",
                    "subILP_bias_fc",
                ]
            elif double_weighted:
                params["fig_width"] = 27.0
                params["specific_models"] = [
                    "subILP_bias",
                    "subILP_bias_weighted_features",
                    "corr_thresh_bias",
                    "corr_thresh_bias_weighted_features",
                ]
            elif less_features:
                params["fig_width"] = 17.0
                params["specific_models"] = [
                    "subILP_bias",
                    "subILP_bias_small",
                ]
            elif original:
                params["specific_models"] = [
                    "rf",
                    "rf_all_features",
                    "corr_thresh",
                    "corr_thresh_all_features",
                ]
            else:
                if regression:
                    params["specific_models"] = [
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
                        params["fig_width"] = 29.0
                        params["specific_models"].extend(
                            [
                                "targeted_subILP_bias_sauron",
                                "targeted_subILP_sauron",
                                "targeted_subILP_bias",
                                "targeted_subILP",
                            ]
                        )
                else:
                    params["fig_width"] = 17.0
                    params["specific_models"] = [
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
                        params["fig_width"] = 19.0
                        params["specific_models"].extend(
                            [
                                "targeted_subILP_bias",
                                "targeted_subILP",
                            ]
                        )
        else:
            continue
        sub_analysis.append((analysis_type, params))
    return sub_analysis
