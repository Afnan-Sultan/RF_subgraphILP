import json
import re

import pandas as pd


def process_trees_info(file):
    trees_info = []
    with open(file, "r", encoding="utf-8") as f:
        json_data = re.sub(r"}\s*{", "},{", f.read())
        trees_info.extend(json.loads("[" + json_data + "]"))

    trees_info = {
        list(e.keys())[0]: {
            "cell_lines": e[list(e.keys())[0]]["cell_lines"],
            "features": e[list(e.keys())[0]]["features"],
            "features_importance": e[list(e.keys())[0]]["features_importance"],
        }
        for e in trees_info
    }
    used_features = {}
    for _, val in trees_info.items():
        for idx, feature in enumerate(val["features"]):
            if feature in used_features:
                used_features[feature]["count"] += 1
                used_features[feature]["importance"] += val["features_importance"][idx]
            else:
                used_features[feature] = {
                    "count": 1,
                    "importance": val["features_importance"][idx],
                }

    used_features = pd.DataFrame(used_features).transpose()
    return used_features


if __name__ == "__main__":
    file_ = "/home/afnan/Afnan/Thesis/RF_subgraphILP/results/zscore_classification_weighted_biased_1_gt_800/intermediate_output/5-Fluorouracil/rf_tress_info/subgraphilp_trees_indices_features.json"
    used_feats = process_trees_info(file_)
