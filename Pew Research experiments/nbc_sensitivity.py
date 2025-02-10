import numpy as np
import pandas as pd
from utils.utils import *
from utils.constants import *
from tqdm import tqdm


def plot_cdf(tv_distances):
    """
    Plot the CDF of the TV distances.
    """
    plt.rcdefaults()
    x = np.linspace(0, 1, 1000)
    cdf = [(tv_distances < xx).mean() for xx in x]

    fig = figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(
        x,
        cdf,
    )  # Clean, bold line
    ax1.set_xlabel(
        "TV distance from uniform",
    )
    ax1.set_ylabel(
        "CDF",
    )
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)
    plt.grid(alpha=0.3)
    subplots_adjust()
    plt.savefig("cdf_order_change.pdf", dpi=300)
    plt.show()


def find_alternative(
    data,
    meta,
    survey_name,
    question_label,
    type_columns,
    refused_label,
    refused_answers,
):
    """
    Find the alternative distribution for a given question that flips the order, if any
    """
    weight_column = f"WEIGHT_{survey_name}"
    demo_columns = meta.demographics

    validate_columns(type_columns, demo_columns)
    data_q = data[[question_label]]
    weights = data[weight_column].values
    condition = filter_responses(
        data, type_columns, refused_label, question_label, refused_answers
    )
    types, type_mappers, unique_types_labels = process_types(
        data, type_columns, condition
    )
    weight_ = weights[condition]
    p_types = calculate_type_probabilities(types, weight_)
    responses = np.array(data_q.values)[condition, :]
    options = {
        i: np.unique(responses[~np.isnan(responses[:, i]), i])
        for i in range(responses.shape[1])
    }
    _, rewards_per_type = calculate_probabilities(
        responses,
        weight_,
        types,
        np.unique(types),
        p_types,
        options=options,
    )

    options = [meta.variable_value_labels[question_label][i] for i in options[0]]
    # Dynamically generate option aliases
    if len(options) <= 2:
        print("only one option")
        return None, len(options)

    if len(responses) / len(options) < 100:
        print(f"small ratio {len(responses)/ len(options)}")
        return None, len(options)

    avg_nbc = np.concatenate(
        [
            calculate_nbc(
                rewards_per_type[i, :, :],
                np.array(p_types),
                np.ones(len(options)) / len(options),
            ).reshape(1, -1)
            for i in range(rewards_per_type.shape[0])
        ],
        axis=0,
    )
    # can we flip the first two options?
    k2, k1 = np.argsort(avg_nbc.mean(0))[-2:]
    k1 = np.argmax(avg_nbc.mean(0))
    funs = []
    ordered_indices = np.argsort(avg_nbc.mean(0))[::-1]
    print(ordered_indices)
    for n, k1 in enumerate(ordered_indices):
        for k2 in ordered_indices[n + 1 :]:
            print(k1, k2)
            if k1 == k2:
                continue
            _, fun = find_alt_dist(
                rewards_per_type[0, :, :],
                np.array(p_types),
                k1,
                k2,
                delta=1e-5,
                epsilon=1e-5,
            )
            if fun is not None:
                funs.append(fun)
    if len(funs) >= 1:
        return min(funs), len(options)
    else:
        return None, len(options)


# call find_alternative for all questions in all surveys
tv_distances = []
sizes = []
c = 0
fail = 0
type_columns = ["F_PARTYSUM_FINAL"]

for survey_name, file_path in SURVEYS.items():
    data, meta = read_data(file_path, survey_name)
    refused_label = 99
    refused_answers = [99]
    for k in tqdm(range(len(meta.questions))):
        question_label = meta.questions[k]
        try:
            tv_distance, options_len = find_alternative(
                data,
                meta,
                survey_name,
                question_label,
                type_columns,
                refused_label,
                refused_answers,
            )
            if tv_distance is not None:
                tv_distances.append(tv_distance)
                sizes.append(options_len)
            if options_len > 2:
                c += 1
        except Exception as e:
            print(e, "fail")
            tv_distance = None
            fail += 1
# print the percentage of questions that can be flipped
print(len(tv_distances) / c)

plot_cdf(tv_distances)
