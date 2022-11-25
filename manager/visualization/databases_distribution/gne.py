"""
Three files were downloaded from the publications visualization tables at the following link
https://www.nature.com/articles/nbt.3080#Sec27

1- cell information, visualization table 1 ("Cell line" and "Tissue Diagnosis" columns were stored to csv file)
2- drug response for Crizotinib, visualization table 11 ("Molecule" and "Cell line" columns were stored to a csv)
3- drug response for 5 drugs, visualization table 13 (all cells were stored to csv file after replacing "NA" with NaN
to be recognized by pandas)
"""

import pandas as pd


def process_gne(print_analysis=False):
    print("Processing GNE")
    drugs_info_5 = pd.read_csv("gne_5_drugs.csv", index_col=0)
    drug_info_1 = pd.read_csv("gne_1_drug.csv")
    cl_per_drugs = drugs_info_5.count()
    cl_per_drugs[drug_info_1.iloc[1, 0]] = len(drug_info_1)
    cl_per_drugs.to_csv("cl_per_drug_gne.csv")
    if print_analysis:
        print("number of drugs = ", len(cl_per_drugs))
        print("average cell lines per drugs = ", cl_per_drugs.mean())
        print("standard deviation cell lines per drugs = ", cl_per_drugs.std())

    cl_meta = pd.read_csv("gne_cell_info.csv")
    cl_of_drugs = set(drugs_info_5.index.to_list())
    cl_of_drugs.update(drug_info_1["Cell line"])
    cl = cl_meta[cl_meta["Cell line"].isin(cl_of_drugs)]
    cl_per_type = cl.groupby("Tissue Diagnosis").count()["Cell line"]
    cl_per_type.to_csv("cl_per_type_gne.csv")
    if print_analysis:
        print("\nnumber of cell lines = ", len(cl))
        print("number of cancer types = ", len(cl_per_type))
        print("average cell lines per cancer type =", cl_per_type.mean())
        print("standard deviation cell lines per cancer type =", cl_per_type.std())
