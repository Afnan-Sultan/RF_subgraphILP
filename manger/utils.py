import json

import numpy as np
import pandas as pd


def get_thresh(drug_name: str, thresholds: pd.DataFrame):
    """
    given a drug name and a threshold DataFrame, search ofr the drug in the df.
    drug name can be formulated as name___screenID
    threshold.columns = ["Drug_ID", "Threshold"]
    """
    name_split = drug_name.split("___")
    if len(name_split) > 1:
        screen_name = name_split[0]
        screen_id = name_split[1]
        drug_thresh_df = thresholds.loc[screen_name]
        thresh = drug_thresh_df[drug_thresh_df["Drug_ID"] == int(screen_id)].loc[
            screen_name, "Threshold"
        ]
    else:
        thresh = thresholds.loc[drug_name, "Threshold"]
    return thresh


class NewJsonEncoder(json.JSONEncoder):
    """
    overrides json encoder to be able to serialize numpy arrays, pandas dataframes, and get the parameters for an
    estimator.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            return obj.to_json(orient="split")
        elif isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)
