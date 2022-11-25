import os
import shutil

result_dirs = [
    "/local/asultan/RF_subgraphILP/results_larger/regression_weighted_biased_gt_750/intermediate_output",
    "/local/asultan/RF_subgraphILP/results_larger/classification_weighted_biased_gt_750/intermediate_output",
]
for result_dir in result_dirs:
    drugs = os.listdir(result_dir)
    dest_folder = os.path.join("/".join(result_dir.split("/")[:-1]), "rf_trees_info")
    for drug in drugs:
        curr_folder = os.path.join(result_dir, drug, "rf_trees_info")
        new_folder = os.path.join(dest_folder, drug)
        shutil.copytree(curr_folder, new_folder)
