import random
import warnings

import pandas as pd


def random_samples(features, num_features, n_rand):
    rand_features = []
    for ctr in range(n_rand):
        sample_features = random.sample(features, num_features)
        rand_features.append(sample_features)
    return rand_features


def get_num_features(num_features_file, drug_name):
    num_features_info = pd.read_csv(num_features_file, index_col=0)
    num_features = num_features_info.loc[drug_name, "num_features"]
    if not isinstance(num_features, int):
        warnings.warn(
            f"{len(num_features)} occurrences of subgraph num_features were found. "
            f"The last instance is used."
        )
        num_features = num_features["num_features"].to_list()[-1]
    return num_features
