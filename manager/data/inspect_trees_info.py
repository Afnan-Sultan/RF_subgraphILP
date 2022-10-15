import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("TkAgg")


def process_trees_info(rf_trees_file):
    trees_json = []
    with open(rf_trees_file, "r", encoding="utf-8") as f:
        json_data = re.sub(r"}\s*{", "},{", f.read())
        trees_json.extend(json.loads("[" + json_data + "]"))

    # fetch each tree index and information
    trees_info = {}
    for tree in trees_json:
        for tree_idx, info in tree.items():  # one element loop
            trees_info[tree_idx] = {
                "cell_lines": info["cell_lines"],
                "features": info["features"],
                "features_importance": info["features_importance"],
            }

    used_features_count = {}
    used_features_importance = {}
    for tree_info in trees_info.values():
        for idx, feature in enumerate(tree_info["features"]):
            if feature in used_features_count:
                used_features_count[feature] += 1
                used_features_importance[feature].append(
                    tree_info["features_importance"][idx]
                )
            else:
                used_features_count[feature] = 1
                used_features_importance[feature] = [
                    tree_info["features_importance"][idx]
                ]

    # used_features = pd.DataFrame(used_features).transpose()
    return used_features_count, used_features_importance


if __name__ == "__main__":
    folders = [
        "../../results_larger/classification_weighted_biased_gt_750/rf_trees_info",
        "../../results_larger/regression_weighted_biased_gt_750/rf_trees_info",
    ]
    trees_features_summary = {}
    for folder in folders:
        ml_method = folder.split("/")[-2].split("_")[0]
        trees_features_summary[ml_method] = {}
        for drug_name in os.listdir(folder):
            trees_features_summary[ml_method][drug_name] = {}
            trees_folder = os.path.join(folder, drug_name)
            for model_trees in os.listdir(trees_folder):
                model = model_trees.split(".")[0]
                trees_file = os.path.join(trees_folder, model_trees)
                counts, importance = process_trees_info(trees_file)
                trees_features_summary[ml_method][drug_name][model] = {
                    "features_counts": pd.DataFrame(counts, index=["count"])
                    .transpose()
                    .sort_values(by="count", ascending=False)
                    .reset_index()
                    .rename({"index": "GeneID"}),
                    "features_importance": importance,
                }

    plot = False
    if plot:
        for ml, drugs in trees_features_summary.items():
            for drug, models in drugs.items():
                fig_idx = 0
                plt.subplots_adjust(wspace=0.5)
                for model, summary in models.items():
                    ax = plt.subplot(len(models) // 3, 4, fig_idx + 1)
                    trees_features_summary[ml][drug][model]["features_counts"].plot(
                        ax=ax, title=model, figsize=(18, 5)
                    )
                    ax.set_xlabel("number of features")
                    ax.set_ylabel(f"number of trees containing features")
                    ax.get_legend().remove()

                    fig_idx += 1

                plt.savefig(
                    f"../../../temp/{ml}/{drug}.png",
                    bbox_inches="tight",
                )
                plt.close()
