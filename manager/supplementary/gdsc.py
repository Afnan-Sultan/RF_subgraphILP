"""
The drugs' metadata file was exported as csv file from the following link
https://www.cancerrxgene.org/compounds

The cell lines metadata file was exported as csv file from the following link
https://www.cancerrxgene.org/celllines
"""

import pandas as pd

cl_meta = pd.read_csv("gdsc_cell_lines_metadata.csv")
drug_meta = pd.read_csv("gdsc_drugs_cell_lines_count.csv")

for version in ["GDSC1"]:  # , "GDSC2"]:
    drugs = drug_meta[drug_meta[" Datasets"] == version][['Drug Id', ' number of cell lines']]
    drugs.columns = ["drug", "num_cell_lines"]
    drugs.to_csv("cl_per_drug_gdsc.csv", index=False)
    print(version)
    print("number of drugs = ", len(drugs))
    print("average cell lines per drugs = ", drugs["num_cell_lines"].mean())
    print("standard deviation cell lines per drugs = ", drugs["num_cell_lines"].std())

    cl = cl_meta[cl_meta[" Datasets"] == version][['Cell line Name', ' TCGA Classfication']]
    cl.columns = ["cell_line", "tumor_type"]
    cl_per_type = cl.groupby("tumor_type").count()["cell_line"]
    cl_per_type.to_csv("cl_per_type_gdsc.csv")
    print("\nnumber of cell lines = ", len(cl))
    print("number of cancer types = ", len(cl_per_type))
    print("average cell lines per cancer type =", cl_per_type.mean())
    print("standard deviation cell lines per cancer type =", cl_per_type.std(), "\n")
