"""
Data file obtained from the publication at the following link (Supplementary Data)
https://www.nature.com/articles/nature17987#Sec25

Table S2 provides information about drug response (columns "drug", "cell_lines" and "gCSI IC50" were stored to csv file)
Table S1 provides information about cell lines, however, only tissue type is reported and not tumor type. Nevertheless,
the cell lines are shared with CCLE and GDSC databases. Hence, reporting the tumor type using these two databases.
"""

import re

import pandas as pd


def process_gsci(print_analysis=False):
    print("Processing gCSI")
    gcsi_drug_response = pd.read_csv("gcsi_drug_response.csv").dropna()
    gcsi_drug_response["cell_line"] = gcsi_drug_response["cell_line"].str.lower()
    cl_per_drugs = gcsi_drug_response.groupby("drug").count()["cell_line"]
    cl_per_drugs.to_csv("cl_per_drug_gcsi.csv")
    if print_analysis:
        print("number of drugs = ", len(cl_per_drugs))
        print("average cell lines per drugs = ", cl_per_drugs.mean())
        print("standard deviation cell lines per drugs = ", cl_per_drugs.std())

    # find cell lines' tumor type from the other databases
    ccle_cell_lines = pd.read_csv("ccle_cell_lines_metadata.csv")
    ccle_cell_lines.columns = ["cell_line", "tumor_type"]
    ccle_cell_lines["cell_line"] = ccle_cell_lines["cell_line"].str.lower()

    gdsc_cell_lines = pd.read_csv("gdsc_cell_lines_metadata.csv")[
        ["Cell line Name", " TCGA Classfication"]
    ]
    gdsc_cell_lines.columns = ["cell_line", "tumor_type"]
    gdsc_cell_lines["cell_line"] = gdsc_cell_lines["cell_line"].str.lower()

    # find intersections between gCSI and the two databases
    gcsi_gdsc = pd.merge(
        gdsc_cell_lines, gcsi_drug_response[["cell_line"]], on="cell_line"
    ).drop_duplicates()
    gcsi_ccle = pd.merge(
        ccle_cell_lines, gcsi_drug_response[["cell_line"]], on="cell_line"
    ).drop_duplicates()

    # find the intersection between the two databases and the gCSI cell lines, then use only one dataset to store the
    # tumor types of the common cell lines to avoid different namings
    ccle_gdsc_intersection = set(gcsi_gdsc["cell_line"].to_list()).intersection(
        gcsi_ccle["cell_line"]
    )
    gdsc_only = set(gcsi_gdsc["cell_line"]) - ccle_gdsc_intersection
    gcsi_cell_lines_meta_data = pd.concat(
        [gcsi_ccle, gcsi_gdsc[gcsi_gdsc["cell_line"].isin(gdsc_only)]]
    )

    # some cell lines were missing, which turned out to be for punctuation variations. The below conditions were
    # successful to restore most of the missing types (only (7 out of 426) cell lines remained unidentified)
    missing = set(gcsi_drug_response["cell_line"]) - set(
        gcsi_cell_lines_meta_data["cell_line"]
    )
    temp_1 = {drug.replace(" ", "-"): drug for drug in missing}
    temp_2 = {drug.replace("-", " "): drug for drug in missing}
    temp_3 = {drug.replace("-", ""): drug for drug in missing}
    temp_4 = {drug.replace(" ", ""): drug for drug in missing}
    temp_4_1 = {drug.replace(" ", "").capitalize(): drug for drug in missing}
    temp_5 = {
        "-".join(re.split(r"(\d+)", drug.replace("-", "")))[:-1].replace(" ", ""): drug
        for drug in missing
    }
    temp_6 = {drug.replace("ii", "-ii"): drug for drug in missing}
    for temp in [temp_1, temp_2, temp_3, temp_4, temp_4_1, temp_5, temp_6]:
        gcsi_ccle_2 = set(temp.keys()).intersection(ccle_cell_lines["cell_line"])
        gcsi_gdsc_2 = (
            set(temp.keys()).intersection(gdsc_cell_lines["cell_line"]) - gcsi_ccle_2
        )
        for drug in gcsi_ccle_2:
            missing.discard(temp[drug])
        for drug in gcsi_gdsc_2:
            missing.discard(temp[drug])
        gcsi_cell_lines_meta_data = pd.concat(
            [
                gcsi_cell_lines_meta_data,
                ccle_cell_lines[ccle_cell_lines["cell_line"].isin(gcsi_ccle_2)],
                gdsc_cell_lines[gdsc_cell_lines["cell_line"].isin(gcsi_gdsc_2)],
            ]
        )

    cl_per_type = gcsi_cell_lines_meta_data.groupby("tumor_type").count()["cell_line"]
    cl_per_type.to_csv("cl_per_type_gcsi.csv")
    if print_analysis:
        print("\nnumber of cell lines = ", len(gcsi_cell_lines_meta_data))
        print("number of cancer types = ", len(cl_per_type) + len(missing))
        print("average cell lines per cancer type =", cl_per_type.mean())
        print("standard deviation cell lines per cancer type =", cl_per_type.std())
