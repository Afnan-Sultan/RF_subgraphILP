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
    # TODO: Inquire --> the simple weight makes the sum of both sen and res equal.
    #  Moreover, the resulting values are too small ( < 0.0x for sens and < 0.00x for res). Is this what we need to do?
    weights = get_weights(train_scores, kwargs)
    sum_weights_sens = 0
    sum_weights_res = 0
    for i in range(len(weights)):
        if classes.iloc[i] == 0:
            sum_weights_res += weights[i]
        elif classes.iloc[i] == 1:
            sum_weights_sens += weights[i]
        else:
            raise ValueError

    for i in range(len(weights)):
        if classes.iloc[i] == 0:
            weights[i] = weights[i] / sum_weights_res
        elif classes.iloc[i] == 1:
            weights[i] = weights[i] / sum_weights_sens
        else:
            raise ValueError
    return train_features.mul(weights, axis=0)
