import pandas as pd
from manager.training.subgraphilp import get_samples


def calculate_linear_weights(threshold, response_values):
    sum_weights_sensitive = 0.0
    sum_weights_resistant = 0.0
    for value in response_values:
        if value < threshold:
            sum_weights_sensitive = sum_weights_sensitive + abs(value - threshold)
        else:
            sum_weights_resistant = sum_weights_resistant + abs(value - threshold)

    weights = []
    for value in response_values:
        if value < threshold:
            current_weight = abs(value - threshold) / (2 * sum_weights_sensitive)
            weights.append(current_weight)
        else:
            current_weight = abs(value - threshold) / (2 * sum_weights_resistant)
            weights.append(current_weight)
    return weights


def calculate_simple_weights(threshold, response_values):
    sum_weights_sensitive = 0.0
    sum_weights_resistant = 0.0
    for value in response_values:
        if value < threshold:
            sum_weights_sensitive = sum_weights_sensitive + 1
        else:
            sum_weights_resistant = sum_weights_resistant + 1

    weights = []
    if sum_weights_sensitive < sum_weights_resistant:
        factor = sum_weights_resistant / sum_weights_sensitive
        sens_less = True
    else:
        factor = sum_weights_sensitive / sum_weights_resistant
        sens_less = False

    for value in response_values:
        if sens_less:
            if value < threshold:
                current_weight = factor
                weights.append(current_weight)
            else:
                current_weight = 1
                weights.append(current_weight)
        else:
            if value < threshold:
                current_weight = 1
                weights.append(current_weight)
            else:
                current_weight = factor
                weights.append(current_weight)
    return weights


def get_weights(train_scores, kwargs):
    if kwargs.training.weight_samples:
        if kwargs.training.simple_weight:
            return calculate_simple_weights(kwargs.data.drug_threshold, train_scores)
        else:
            return calculate_linear_weights(kwargs.data.drug_threshold, train_scores)
    else:
        return None


def weighted_expression(train_features, train_scores, classes, kwargs):
    # multiply all genes of a sample by its weight
    weights = get_weights(train_scores, kwargs)
    return train_features.mul(weights, axis=0)
    # weighted_mat = train_features.mul(weights, axis=0)
    #
    # # separate sensitive from resistant cell lines
    # mat_per_label = get_samples(
    #     weighted_mat, classes, {"ref": 0, "sample": 1}, cls_as_cols=False
    # )
    # ref, sample = mat_per_label["ref"], mat_per_label["sample"]
    #
    # # divide each gene by the sum of its expression in the corresponding class
    # ref_summed = ref.div(ref.sum(axis=0), axis=1)
    # sample_summed = sample.div(sample.sum(axis=0), axis=1)
    #
    # # concatenate the two classes after processing. assert columns order is the same to ensure correct concatenation
    # assert all(ref_summed.columns == sample_summed.columns)
    # weighted_summed_mat = pd.concat([ref_summed, sample_summed])
    #
    # # reorder cell lines as originally passed. Just a cautious step.
    # weighted_summed_mat = weighted_summed_mat.loc[train_features.index]
    # return weighted_summed_mat
