"""
Data downloaded from the publication's supplementary material at the following link (Supplementary Table 1)
https://www.nature.com/articles/nm.3954#Sec28

3 columns from the "PCT raw data" sheet were collected ("Model", "Tumor Type", "Treatment") and stored to a csv file

"""

import pandas as pd

pdx_info = pd.read_csv("nibr_pdxe_data.csv")

pdx_drugs = pdx_info[["Model", "Treatment"]].drop_duplicates()
pdx_per_drugs = pdx_drugs.groupby("Treatment").count()["Model"]
pdx_per_drugs.to_csv("pdx_per_drug_pdxe.csv")
print("number of drugs = ", len(pdx_per_drugs))
print("average cell lines per drugs = ", pdx_per_drugs.mean())
print("standard deviation cell lines per drugs = ", pdx_per_drugs.std())

pdx_meta = pdx_info[["Model", "Tumor Type"]].drop_duplicates()
pdx_per_type = pdx_meta.groupby("Tumor Type").count()["Model"]
pdx_per_type.to_csv("pdx_per_type_pdxe.csv")
print("\nnumber of cell lines = ", len(pdx_meta))
print("number of cancer types = ", len(pdx_per_type))
print("average cell lines per cancer type =", pdx_per_type.mean())
print("standard deviation cell lines per cancer type =", pdx_per_type.std())
