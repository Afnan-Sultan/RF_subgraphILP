import random

import pandas as pd


def random_samples(features, num_features, n_rand):
    rand_features = []
    for ctr in range(n_rand):
        sample_features = random.sample(features, num_features)
        rand_features.append(sample_features)
    return rand_features


def calc_corr(train_features: pd.DataFrame, train_scores: pd.Series):
    train_features["scores"] = train_scores.values
    corr_mat = train_features.corr()
    corr_scores = corr_mat["scores"].sort_values().abs().drop(labels=["scores"])
    return corr_scores


def get_num_features(num_features_file, drug_name):
    num_features_info = pd.read_csv(num_features_file, index_col=0)
    num_features = num_features_info.loc[drug_name, "num_features"]
    return num_features
