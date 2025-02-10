import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # For setting nicer styles
import os
import string
from utils.utils import *
from utils.constants import *

# check that all the files above exist, if not raise an error
for survey in SURVEYS:
    if not os.path.exists(SURVEYS[survey]):
        raise FileNotFoundError(
            f"File {SURVEYS[survey]} not found. Make sure to download the file from PEW research website and add\
               it in the surveys directory as indicated in the SURVEYS dictionary."
        )


# Set Seaborn style for clean plots
sns.set_context("talk", font_scale=1.2)
sns.set_style("whitegrid")


def plot(
    rewards,
    probabilities,
    norm_avg_reward,
    norm_avg_nbc,
    labels_types,
    options,
    options_alias,
    keys,
    p_types,
    question_label="",
):
    """
    Generate publication-ready, color-coded plots with improved aesthetics. The rewards and probabilities
    are of shape (num_estimates, num_types, num_options) and quintiles or similar statistics are plotted.
    """

    n_types = len(labels_types)
    x = np.arange(len(options))  # the label locations
    width = 0.8 / n_types  # bar width divided by the number of types
    # Create subplots
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1, 1]}
    )

    # Plot rewards per Type (using quintiles)
    for k, label in enumerate(labels_types):
        values = rewards[:, k, :].mean(0)
        axes[1].bar(
            x + k * width, values, width=width, color=COLOR_PALETTE[k], label=f"{label}"
        )

    axes[1].set_title("Rewards per Type", fontsize=TITLE_FONT_SIZE, fontweight="bold")
    axes[1].set_ylabel("Reward", fontsize=AXIS_LABEL_FONT_SIZE)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].set_xticks(x + width * (n_types - 1) / 2)
    axes[1].set_xticklabels(options_alias, fontsize=AXIS_TICKS_FONT_SIZE)

    # Plot probabilities per Type (using quintiles)
    for k, label in enumerate(labels_types):
        axes[0].bar(
            x + k * width,
            probabilities[:, k, :].mean(0),  # Plot the median (50th percentile)
            width=width,
            label=f"{label}",
            color=COLOR_PALETTE[k],
        )
    axes[0].set_title(
        "Response Frequency per Type", fontsize=TITLE_FONT_SIZE, fontweight="bold"
    )
    axes[0].set_ylabel("Proportion of Responses", fontsize=AXIS_LABEL_FONT_SIZE)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].set_xticks(x + width * (n_types - 1) / 2)
    axes[0].set_xticklabels(options_alias, fontsize=AXIS_TICKS_FONT_SIZE)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.53, -0.2),
        ncol=3,
        title="Types",
        fontsize=26,
        title_fontsize=30,
        frameon=True,
    )

    # Adjust layout to avoid overlap
    plt.tight_layout()  # Adjust for legend
    plt.savefig(
        f"prop_rew_{question_label}.pdf", dpi=1000, bbox_inches="tight"
    )  # PDF format

    # Plot Rewards per Type (single plot for comparison)
    plt.figure(figsize=(12, 5))
    for k, label in enumerate(labels_types):
        plt.bar(
            x + k * width,
            rewards[:, k, :].mean(0),  # Plot the mean (50th percentile)
            width=width,
            label=f"{label}",
            color=COLOR_PALETTE[k],
        )

    plt.title("Rewards per Type", fontsize=TITLE_FONT_SIZE, fontweight="bold")
    plt.ylabel("Reward", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(
        x + width * (n_types - 1) / 2, options_alias, fontsize=AXIS_TICKS_FONT_SIZE
    )
    plt.legend(
        ncol=3, fontsize=20, title_fontsize=22, framealpha=0.6, loc="lower right"
    )
    plt.tight_layout()
    plt.savefig(
        f"rew_{question_label}.pdf", dpi=1000, bbox_inches="tight"
    )  # PDF format
    # Comparison of avg_reward and avg_nbc (Side-by-Side Bar Chart)
    plt.figure(figsize=(12, 5))
    x = np.arange(len(options))  # the label locations
    width = 0.35  # width of the bars

    plt.bar(
        x - width / 2,
        norm_avg_reward.mean(0),
        width,
        label="Avg Reward",
        color=COLOR_PALETTE[0],
    )
    plt.bar(
        x + width / 2, norm_avg_nbc.mean(0), width, label="NBC", color=COLOR_PALETTE[1]
    )
    plt.xticks(x, options_alias, fontsize=AXIS_TICKS_FONT_SIZE)
    plt.yticks(fontsize=AXIS_TICKS_FONT_SIZE)
    plt.legend(
        fontsize=AXIS_TICKS_FONT_SIZE,
        ncol=1,  # Arrange legend entries in 3 columns
        frameon=True,  # Add a frame for clarity
        borderpad=1,  # Add padding inside the frame
    )
    plt.xlabel("Options", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Values", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylim([min(np.min(norm_avg_reward), np.min(norm_avg_nbc)) - 0.1, None])
    plt.tight_layout()
    plt.savefig(
        f"reward_vs_NBC_{question_label}.pdf", dpi=1000, bbox_inches="tight"
    )  # PDF format

    # Ranking Comparison (Bump Chart)
    data = pd.DataFrame(
        {
            "Option": options_alias,
            "Avg_Reward": norm_avg_reward.mean(0),
            "Avg_NBC": norm_avg_nbc.mean(0),
        }
    )
    data["Rank_Reward"] = data["Avg_Reward"].rank(ascending=False).astype(int)
    data["Rank_NBC"] = data["Avg_NBC"].rank(ascending=False).astype(int)

    plt.figure(figsize=(4, 10))
    for i in range(len(data)):
        plt.plot(
            [0.3, 1],
            [data["Rank_Reward"][i], data["Rank_NBC"][i]],
            marker="o",
            label=data["Option"][i],
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            linewidth=5,
        )
        plt.text(
            0.15,
            data["Rank_Reward"][i],
            data["Option"][i],
            fontsize=27,
            va="center",
            fontweight="bold",
        )
        plt.text(
            1.05,
            data["Rank_NBC"][i],
            data["Option"][i],
            fontsize=27,
            va="center",
            fontweight="bold",
        )

    plt.xticks([0.3, 1], ["Avg Reward", " NBC"], fontsize=26)
    plt.gca().invert_yaxis()
    # Set integer y-axis ticks
    y_ticks = plt.gca().get_yticks()  # Get current y-ticks
    plt.gca().set_yticks(
        range(int(min(y_ticks)) + 1, int(max(y_ticks)) + 1)
    )  # Set integer ticks
    plt.gca().set_yticklabels(
        range(int(min(y_ticks)) + 1, int(max(y_ticks)) + 1),
        fontsize=AXIS_TICKS_FONT_SIZE,
    )  # Ensure labels are integers

    plt.xlim([0.1, 1.2])
    plt.xlabel("Metric", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Rank", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        f"ranking_comparison_{question_label}.pdf", dpi=1000, bbox_inches="tight"
    )  # PDF format

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
    # extract and validate data
    weight_column = f"WEIGHT_{survey_name}"
    demo_columns = meta.demographics
    validate_columns(type_columns, demo_columns)
    data_q = data[[question_label]]
    weights = data[weight_column].values
    # filter responses based on refused labels and missing data
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
    # calculate probabilities and rewards
    question_probabilities_per_type, rewards_per_type = calculate_probabilities(
        responses,
        weight_,
        types,
        np.unique(types),
        p_types,
        options=options,
    )
    n_types = len(p_types)
    options = [meta.variable_value_labels[question_label][i] for i in options[0]]
    # Dynamically generate option aliases
    options_alias_dict = dict(zip(options, string.ascii_uppercase[: len(options)]))
    options_alias = [options_alias_dict[op] for op in options]
    keys = list(options[0])

    if len(options[0]) <= 1:
        pass
    else:
        # rewards_per_type is of size (1, no_types, no_options)
        avg_reward = np.concatenate(
            [(rewards_per_type[:, t : t + 1, :]) * p_types[t] for t in range(n_types)],
            axis=1,
        ).sum(1)

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

    labels_types = extract_labels(unique_types_labels, type_columns, meta)

    plot(
        rewards_per_type,
        question_probabilities_per_type,
        avg_reward,
        avg_nbc,
        labels_types,
        options,
        options_alias,
        keys,
        p_types,
        question_label,
    )
    return question_probabilities_per_type, rewards_per_type, p_types


# examples in the paper
question_labels = [
    "COVIDEGFP_a_W114",
    "FAVPOL_BIDEN_W130",
    "FAVPOL_HARRIS_W130",
    "EVCAR2_W128",
    "BIDENADM_W83",
]
if __name__ == "__main__":
    for question_label in question_labels:
        survey_name = question_label.split("_")[-1]
        type_columns = ["F_PARTYSUM_FINAL"]
        file_path = SURVEYS[survey_name]
        data, meta = read_data(file_path, survey_name)
        refused_label = 99
        refused_answers = [99]
        probabilities, rewards, p_types = main(
            data,
            meta,
            survey_name,
            question_label,
            type_columns,
            refused_label,
            refused_answers,
        )
