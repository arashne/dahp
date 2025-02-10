import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For setting nicer styles
import string
from utils.utils import *
from utils.constants import *

# check that all required files exist, if not raise an error
check_files_exist()

# Set Seaborn style for clean plots
sns.set_context("talk", font_scale=1.2)
sns.set_style("whitegrid")


def plot(
    norm_avg_reward, norm_avg_nbc, options_alias, question_label="", avg_nbc_alt=None
):
    """
    Generate publication-ready, color-coded plots with improved aesthetics. The rewards and probabilities
    are of shape (num_estimates, num_types, num_options) and quintiles or similar statistics are plotted.
    """

    # Option-to-description mapping for legend
    option_legend = {
        "A": "Very likely",
        "B": "Somewhat likely",
        "C": "Not too likely",
        "D": "Not at all likely",
        "E": "Not purchasing",
    }

    # Ranking Comparison (Transposed Bump Chart)
    data = pd.DataFrame(
        {
            "Option": options_alias,
            "Avg_Reward": norm_avg_reward.mean(0),
            "Avg_NBC": norm_avg_nbc.mean(0),
            "Avg_NBC_alt": avg_nbc_alt.mean(0),
        }
    )
    data["Rank_Reward"] = data["Avg_Reward"].rank(ascending=False).astype(int)
    data["Rank_NBC"] = data["Avg_NBC"].rank(ascending=False).astype(int)
    data["Rank_NBC_alt"] = data["Avg_NBC_alt"].rank(ascending=False).astype(int)

    plt.figure(figsize=(10, 4))
    for i in range(len(data)):
        plt.plot(
            [data["Rank_Reward"][i], data["Rank_NBC"][i], data["Rank_NBC_alt"][i]],
            [0.3, 1, 1.7],
            marker="o",
            label=f"{data['Option'][i]}: {option_legend.get(data['Option'][i], 'Unknown')}",
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            linewidth=5,
        )
        plt.text(
            data["Rank_Reward"][i],
            0.15,
            data["Option"][i],
            fontsize=27,
            ha="center",
            fontweight="bold",
        )
        plt.text(
            data["Rank_NBC"][i],
            1.05,
            data["Option"][i],
            fontsize=27,
            ha="center",
            fontweight="bold",
        )
        plt.text(
            data["Rank_NBC_alt"][i],
            1.6,
            data["Option"][i],
            fontsize=27,
            ha="center",
            fontweight="bold",
        )

    plt.yticks([0.3, 1, 1.7], ["Avg Reward", "NBC ($D_U$)", "NBC ($D_a$)"], fontsize=26)
    # Set integer x-axis ticks
    x_ticks = plt.gca().get_xticks()
    plt.gca().set_xticks(range(int(min(x_ticks)) + 1, int(max(x_ticks)) + 1))
    plt.gca().set_xticklabels(
        range(int(min(x_ticks)) + 1, int(max(x_ticks)) + 1),
        fontsize=AXIS_TICKS_FONT_SIZE,
    )
    plt.ylim([0.1, 1.8])
    plt.ylabel("Metric", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.xlabel("Rank", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)

    # Add legend on top with multiple columns
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(+0.5, 1.5),  # Centered above the plot
        ncol=2,  # Number of columns
        fontsize=AXIS_LABEL_FONT_SIZE - 4,
        frameon=False,
    )

    # plt.tight_layout()
    plt.savefig(
        f"ranking_comparison_sens_{question_label}.pdf", dpi=1000, bbox_inches="tight"
    )
    plt.show()


def main(
    data,
    meta,
    survey_name,
    question_label,
    type_columns,
    refused_label,
    refused_answers,
):
    """
    Main function to orchestrate the entire process.
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
    question_probabilities_per_type, rewards_per_type = calculate_probabilities(
        responses,
        weight_,
        types,
        np.unique(types),
        p_types,
        options=options,
    )

    n_types = len(p_types)
    q = 0
    options = [meta.variable_value_labels[question_label][i] for i in options[0]]
    # Dynamically generate option aliases
    options_alias_dict = dict(zip(options, string.ascii_uppercase[: len(options)]))
    options_alias = [options_alias_dict[op] for op in options]
    keys = list(options[q])

    if len(options) <= 1:
        pass
    else:
        avg_reward = np.concatenate(
            [(rewards_per_type[:, t : t + 1, :]) * p_types[t] for t in range(n_types)],
            axis=1,
        ).sum(1)

    labels_types = extract_labels(unique_types_labels, type_columns, meta)
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
    k2, k1 = np.argsort(avg_nbc.mean(0))[-2:]
    alt_dist, fun = find_alt_dist(rewards_per_type[0, :, :], np.array(p_types), k1, k2)
    if alt_dist is not None:
        avg_nbc_alt = np.concatenate(
            [
                calculate_nbc(
                    rewards_per_type[i, :, :], np.array(p_types), alt_dist
                ).reshape(1, -1)
                for i in range(rewards_per_type.shape[0])
            ],
            axis=0,
        )
    else:
        avg_nbc_alt = None
    plot(avg_reward, avg_nbc, options_alias, question_label, avg_nbc_alt)


if __name__ == "__main__":
    question_label = "EVCAR2_W128"
    survey_name = question_label.split("_")[-1]
    type_columns = ["F_PARTYSUM_FINAL"]
    file_path = SURVEYS[survey_name]
    data, meta = read_data(file_path, survey_name)
    refused_label = 99
    refused_answers = [99]
    main(
        data,
        meta,
        survey_name,
        question_label,
        type_columns,
        refused_label,
        refused_answers,
    )
