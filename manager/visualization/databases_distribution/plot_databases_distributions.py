import os

import matplotlib.pyplot as plt
import pandas as pd

from ccle import process_ccle
from ctrpv2 import process_ctrpv2
from gcsi import process_gsci
from gdsc import process_gdsc
from gne import process_gne
from nci60 import process_nci60
from nibr_pdxe import process_nibr

# extract relevant information from each database. Results summary can be printed to std by passing
# "print_analysis=True"
process_ccle()
process_ctrpv2()
process_gsci()
process_gdsc()
process_gne()
process_nci60()
process_nibr()

print("Plotting results")

# assign proper naming for the databases
database_convert = {
    "ccle": "CCLE",
    "ctrpv2": "CTRPv2",
    "gcsi": "gCSI",
    "gdsc": "GDSC1",
    "gne": "GNE",
    "nci60": "NCI60",
    "pdxe": "NIBR-PDXE",
}

# fetch the files outputted from the databases processing
cl_per_drug_files = [
    file for file in os.listdir(os.getcwd()) if file.startswith("cl_per_drug")
]
cl_per_type_files = [
    file for file in os.listdir(os.getcwd()) if file.startswith("cl_per_type")
]

pdxe_db = {
    "drugs": [
        file for file in os.listdir(os.getcwd()) if file.startswith("pdx_per_drug")
    ][0],
    "cancer_type": [
        file for file in os.listdir(os.getcwd()) if file.startswith("pdx_per_type")
    ][0],
}


# boxplot the results
plt.figure(figsize=(15, 6))
idx_label = {1: "(A)", 2: "(B)", 3: "(C)"}
idx = 1
fontsize = 14
for key, val in {
    "pdxe": pdxe_db,
    "Drugs": cl_per_drug_files,
    "Cancer Type": cl_per_type_files,
}.items():
    if key == "pdxe":
        dist_info = {f"pdx_per_{k}": pd.read_csv(v).iloc[:, 1] for k, v in val.items()}
        ax = plt.subplot(1, 3, idx)
        ax.boxplot(dist_info.values())
        ax.set_xticklabels(dist_info.keys(), fontsize=fontsize)
        ax.set_ylabel("Number of PDXs", fontsize=fontsize - 2)
        ax.set_title(
            f"PDXs Distribution for {database_convert[key]}", fontsize=fontsize
        )
        ax.annotate(
            idx_label[idx],
            xy=(-0.2, ax.get_ylim()[1] + 5),
            annotation_clip=False,
            fontsize=fontsize,
            weight="bold",
        )
        idx += 1
        plt.xticks(rotation=8, ha="center")
        continue

    dist_info = {}
    for file in val:
        database = file.split(".")[0].split("_")[-1]
        info = pd.read_csv(file)
        dist_info[database_convert[database]] = info.iloc[:, 1]

    ax = plt.subplot(1, 3, idx)
    ax.boxplot(dist_info.values())
    ax.set_xticklabels(dist_info.keys(), fontsize=fontsize)
    ax.set_ylabel("Number of Cell Lines", fontsize=fontsize - 2)
    ax.set_title(f"Distribution of Cell Lines per {key}", fontsize=fontsize)
    ax.annotate(
        idx_label[idx],
        xy=(-1.5, ax.get_ylim()[1] + 13),
        annotation_clip=False,
        fontsize=fontsize,
        weight="bold",
    )
    idx += 1
    plt.xticks(rotation=25, ha="center")
plt.subplots_adjust(wspace=0.5)
plt.savefig("db_models_distribution.png")
# plt.show()
