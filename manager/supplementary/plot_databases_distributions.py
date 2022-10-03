import os

import pandas as pd
import matplotlib.pyplot as plt

database_convert = {"ccle": "CCLE",
                    "ctrpv2": "CTRPv2",
                    "gcsi": "gCSI",
                    "gdsc": "GDSC1",
                    "gne": "GNE",
                    "nci60": "NCI60",
                    "pdxe": "NIBR-PDXE"}
cl_per_drug_files = [file for file in os.listdir(os.getcwd()) if file.startswith("cl_per_drug")]
cl_per_type_files = [file for file in os.listdir(os.getcwd()) if file.startswith("cl_per_type")]

pdxe_db = {"drugs": [file for file in os.listdir(os.getcwd()) if file.startswith("pdx_per_drug")][0],
           "cancer_type": [file for file in os.listdir(os.getcwd()) if file.startswith("pdx_per_type")][0]}

plt.figure(figsize=(15, 5))
idx_label = {1: "A", 2: "B", 3: "C"}
idx = 1
for key, val in {"pdxe": pdxe_db, "Drugs": cl_per_drug_files, "Cancer Type": cl_per_type_files}.items():
    if key == "pdxe":
        dist_info = {f"pdx_per_{k}": pd.read_csv(v).iloc[:, 1] for k, v in val.items()}
        ax = plt.subplot(1, 3, idx)
        # fig, ax = plt.subplots()
        ax.boxplot(dist_info.values())
        ax.set_xticklabels(dist_info.keys(), fontsize=12)
        ax.set_ylabel("Number of PDXs")
        ax.set_title(f"PDXs Distribution for {database_convert[key]}")
        ax.annotate(idx_label[idx], xy=(-.2, ax.get_ylim()[1] + 5), annotation_clip=False, fontsize=13)
        idx += 1
        plt.xticks(rotation=5, ha='center')
        continue

    dist_info = {}
    for file in val:
        database = file.split(".")[0].split("_")[-1]
        info = pd.read_csv(file)
        dist_info[database_convert[database]] = info.iloc[:, 1]

    ax = plt.subplot(1, 3, idx)
    # fig, ax = plt.subplots()
    ax.boxplot(dist_info.values())
    ax.set_xticklabels(dist_info.keys(), fontsize=10)
    ax.set_ylabel("Number of Cell Lines")
    ax.set_title(f"Distribution of Cell Lines per {key}")
    ax.annotate(idx_label[idx], xy=(-1.5, ax.get_ylim()[1] + 13), annotation_clip=False, fontsize=13)
    idx += 1
    plt.xticks(rotation=20, ha='center')
plt.subplots_adjust(wspace=0.5)
plt.savefig("db_models_distribution.png")
plt.show()
