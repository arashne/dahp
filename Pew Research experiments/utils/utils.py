import numpy as np
import pyreadstat
from scipy.optimize import linprog
from utils.constants import META_VARS_PREFIX, SURVEYS
import os
import matplotlib.pyplot as plt


def check_files_exist():
    for survey in SURVEYS:
        if not os.path.exists(SURVEYS[survey]):
            raise FileNotFoundError(
                f"File {SURVEYS[survey]} not found. Make sure to download the file from PEW research website and add\
                   it in the surveys directory as indicated in the SURVEYS dictionary."
            )


def validate_columns(type_columns, demo_columns):
    """
    Validate that all type columns exist in the demographics columns.
    """
    assert all(
        [col in demo_columns for col in type_columns]
    ), f"One or more of the type columns are not part of the demo column: {set(type_columns).difference(demo_columns)}"


def filter_responses(
    data, type_columns, refused_label, question_label, refused_answers
):
    """
    Filter responses based on refused labels and missing data.

    Args:
        data (pd.DataFrame): The data to filter.
        type_columns (list): The list of type columns.
        refused_label (int): The refused label.
        question_label (str): The question label.
        refused_answers (list): The refused answers.+
    """
    condition = data[type_columns].apply(
        lambda x: (refused_label not in set(x)) and np.all(~np.isnan(x)), axis=1
    )
    condition = condition & ~(data[question_label].isin(refused_answers))
    return condition


def process_types(data, type_columns, condition):
    """
    Map unique types to integer labels.

    Args:
        data: The data to process.
        type_columns: The type columns.
        condition: The condition to filter the data.

    Returns:
    - types: The integer labels for each type.
    - type_mappers: A dictionary mapping unique types to integer labels.
    - unique_types_labels: The unique types.

    """
    types = data[type_columns].apply(lambda x: ",".join(x.astype(str)), axis=1)
    types = types[condition]
    unique_types_labels = types.unique()
    type_mappers = dict(zip(unique_types_labels, np.arange(len(unique_types_labels))))
    types = types.map(type_mappers)
    return types.astype(int).values, type_mappers, unique_types_labels


def calculate_type_probabilities(types, weights):
    """
    Calculate the probability of each type, weighted by weights.

    Args:
        types (np.array): The types.
        weights (np.array): The weights.

    Returns:
        np.array: The probability of each type.
    """
    counts = np.bincount(types, weights=weights)
    unique_types = np.unique(types)
    p_types = [counts[t] / sum(counts) for t in unique_types]
    return p_types


def calculate_probabilities(
    responses,
    weights,
    types,
    unique_types,
    p_types,
    options,
):
    """
    Calculate question probabilities and rewards for each type.
    """
    num_types = len(p_types)
    question_probabilities_per_type = np.zeros([1, num_types, len(options[0])])
    rewards_per_type = np.zeros([1, num_types, len(options[0])])
    for u, user_type in enumerate(unique_types):
        responses_type = responses[types == user_type, :]
        weights_types = weights[types == user_type]
        probabilities = weighted_column_probabilities(
            responses_type, weights_types, options[0]
        )
        rewards = estimate_rewards_(probabilities)
        question_probabilities_per_type[0, u, :] = [
            probabilities[option] for option in options[0]
        ]
        rewards_per_type[0, u, :] = [rewards[option] for option in options[0]]
    return question_probabilities_per_type, rewards_per_type


def calculate_overall_probabilities(
    responses, weights, options, weighted_column_probabilities
):
    """
    Calculate overall probabilities for all questions.
    """
    return weighted_column_probabilities(responses, weights, options)


# utility functions


def solve_linear_program(P, k1, k2, epsilon=1e-5, delta=1e-5):
    """
    Solve the linear program:
        min (1/2) * sum(s_i) subject to:
            s_i > epsilon
            s_i >= 1/N - q_i
            s_i >= q_i - 1/N
            P[k1, :] @ q < P[k2, :] @ q + delta
            sum(q_i) = 1
            q_i > 0
    """
    N = P.shape[0]  # Size of the square matrix P

    # Decision variables: q and s
    # We have N variables for q and N variables for s
    # Objective function coefficients: minimize (1/2) * sum(s_i)
    c = np.zeros(2 * N)  # First N elements for q, next N for s
    c[N:] = 1 / 2  # Coefficients for s_i

    # Inequality constraints for s_i and q_i
    A_ineq = []
    b_ineq = []

    # Constraints for s_i >= epsilon
    for i in range(N):
        row = np.zeros(2 * N)
        row[N + i] = -1  # Coefficient for s_i
        A_ineq.append(row)
        b_ineq.append(epsilon)

    # Constraints for s_i >= 1/N - q_i
    for i in range(N):
        row = np.zeros(2 * N)
        row[N + i] = -1  # Coefficient for s_i
        row[i] = -1  # Coefficient for q_i
        A_ineq.append(row)
        b_ineq.append(-1 / N)

    # Constraints for s_i >= q_i - 1/N
    for i in range(N):
        row = np.zeros(2 * N)
        row[N + i] = -1  # Coefficient for s_i
        row[i] = 1  # Coefficient for q_i
        A_ineq.append(row)
        b_ineq.append(1 / N)
    # Constraint for P[k1, :] @ q < P[k2, :] @ q + delta
    row = np.zeros(2 * N)
    row[:N] = P[k1, :] - P[k2, :]  # Coefficients for q
    A_ineq.append(row)
    b_ineq.append(-delta)

    # Convert constraints to arrays
    A_ineq = np.array(A_ineq)
    b_ineq = np.array(b_ineq)

    # Equality constraint: sum(q_i) = 1
    A_eq = np.zeros((1, 2 * N))
    A_eq[0, :N] = 1  # Coefficients for q_i
    b_eq = np.array([1])

    # Bounds for q_i and s_i
    bounds = [(epsilon, None)] * N + [(epsilon, None)] * N  # q_i >= 0, s_i >= epsilon

    # Solve the linear program
    result = linprog(
        c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    return result


def find_alt_dist(rewards, p_types, k1, k2, delta=1e-3, epsilon=0):
    P = calculate_pairwise_matrix(rewards, p_types)  # Example matrix P of size NxN

    N = len(P)
    result = solve_linear_program(P, k1, k2, epsilon, delta)

    if result.success:
        print(f"Optimal solution found: {result.fun}")
        q = result.x[:N]
        s = result.x[N:]
        print("q:", q)
        print("s:", s)
        return np.array(q), result.fun
    else:
        print("No solution found:", result.message)
        return None, None


def read_data(file_path, survey_name):
    """
    Reads data from a .sav file and processes metadata.

    Args:
        file_path (str): The path to the .sav file.
        survey_name (str): The name of the survey to be appended to certain metadata variables.

    Returns:
        tuple: A tuple containing:
            - df (pandas.DataFrame): The data from the .sav file.
            - meta (pyreadstat.metadata_container): The metadata from the .sav file, with additional attributes:
                - questions (list): A list of question variable names.
                - demographics (list): A list of demographic variable names.
    """
    df, meta = pyreadstat.read_sav(file_path)
    # add suffix
    meta_vars = []
    for var in META_VARS_PREFIX:
        if var[-1] == "_":
            meta_vars.append(var + survey_name)
        else:
            meta_vars.append(var)
    # Get questions and demographics
    demographics = [var for var in meta.column_names if var.startswith("F_")]
    questions = [
        var
        for var in meta.column_names
        if (var not in meta_vars)
        and (var not in demographics)
        and ("WEIGHT_" not in var)
    ]
    # Add questions and demographics as attributes to the metadata
    meta.__setattr__("questions", questions)
    meta.__setattr__("demographics", demographics)
    return df, meta


def extract_labels(unique_types_labels, type_columns, meta):
    """
    Extract labels for each type.

    Args:
        unique_types_labels (list): List of unique types.
        type_columns (list): List of type columns.
        meta (pyreadstat.metadata_container): Metadata container.

    Returns:
        list: List of labels for each type.
    """
    labels_types = []
    for type_ in unique_types_labels:
        if type_ == "nan":
            labels_types.append("NAN")
        else:
            if not isinstance(type_, str):
                type_ = str(type_)
            label = "_".join(
                [
                    meta.variable_value_labels[type_column][
                        int(float(type_.split(",")[i]))
                    ]
                    for i, type_column in enumerate(type_columns)
                ]
            )
            labels_types.append(label)
    return labels_types


def weighted_column_probabilities(array, weights, options=None):
    """
    Calculate the weighted probabilities of each option in a question.

    Args:
        array (np.array): An array of responses to a question.
        weights (np.array): An array of weights for each response.
        options (list): A list of options for the question.

    Returns:
        dict: A dictionary of probabilities (values) for each option (key).
    """
    # Initialize a dictionary to store probabilities for each column
    col_data = array[:, 0]
    weights_ = weights[~np.isnan(col_data)]
    col_data = col_data[~np.isnan(col_data)]
    if len(col_data) > 0:
        unique, inverse = np.unique(col_data, return_inverse=True)
        weighted_counts = np.bincount(inverse, weights=weights_)
        probs = dict(zip(unique, weighted_counts / weighted_counts.sum()))
    else:
        probs = {}
    if options is not None:
        for opt in options:
            if opt not in probs:
                probs[opt] = 0
    if sum(list(probs.values())) == 0:
        for opt in options:
            probs[opt] = 1 / len(options)
    return probs


def calculate_nbc(rewards):
    """
    Vectorized calculation of the NBC (Normalized Borda Count) score for each option.

    Args:
        rewards (list or np.array): A list or array of reward values for each option.

    Returns:
        np.array: An array of NBC scores for each option.
    """
    rewards = np.array(rewards)
    n_options = len(rewards)

    # Compute the pairwise preference matrix using the Bradley-Terry model
    # PR(A > B) = e^{r_A} / (e^{r_A} + e^{r_B})
    exp_rewards = np.exp(rewards)
    pairwise_matrix = exp_rewards[:, None] / (
        exp_rewards[:, None] + exp_rewards[None, :]
    )
    # Exclude self-comparison by setting diagonal to 0
    np.fill_diagonal(pairwise_matrix, 0)

    if n_options == 1:
        nbc_scores = pairwise_matrix.sum(axis=1)
    else:
        # Sum over each row (preferences of A over all B), and normalize
        nbc_scores = pairwise_matrix.sum(axis=1) / (n_options - 1)

    return nbc_scores


def calculate_pairwise_matrix(rewards, p_u):
    """
    Calculate the pairwise preference matrix using the Bradley-Terry model given a 2d array of rewards (n_types x n_options) and probability of each type (p_u).

    Args:
        rewards (np.ndarray): A 2d array of reward values for each option, for each type.
        p_u (np.ndarray): Probability of each type.

    Returns:
        np.ndarray: A 2d array of pairwise preference probabilities of size (n_options x n_options).
    """
    no_types, n_options = rewards.shape
    # Compute the pairwise preference matrix using the Bradley-Terry model
    # PR(A > B) = e^{r_A} / (e^{r_A} + e^{r_B})
    pairwise_matrix = np.zeros([n_options, n_options])
    for u in range(no_types):
        exp_rewards = np.exp(rewards[u, :])
        pairwise_matrix_u = exp_rewards[:, None] / (
            exp_rewards[:, None] + exp_rewards[None, :]
        )
        # np.fill_diagonal(pairwise_matrix, 0)
        pairwise_matrix += p_u[u] * pairwise_matrix_u
    return pairwise_matrix


def calculate_nbc(rewards, p_u, p_y):
    """
    Calculate the NBC (Normalized Borda Count) score for each option, given a 2d array of rewards (n_types x n_options),
    and probability of each type (p_u) and the probability each option showing up in the pairwise comparison (p_y).

    Args:
        rewards (np.ndarray): A 2d array of reward values for each option, for each type.
        p_u (np.ndarray): Probability of each type.
        p_y (np.ndarray): Probability of each option showing up in the pairwise comparison.

    Raises:
        ValueError: If probabilities do not sum to 1.


    Returns:
        np.array: An array of NBC scores for each option.
    """
    if not np.isclose(p_u.sum(), 1):
        raise ValueError("Probabilities  p_u must sum to 1.")
    if not np.isclose(p_y.sum(), 1):
        raise ValueError("Probabilities p_y  mustsum to 1.")

    matrix = calculate_pairwise_matrix(rewards, p_u)
    nbc = np.dot(matrix, p_y)
    return nbc


def estimate_rewards(probabilities):
    """
    Estimate rewards from given choice probabilities using the Luce-Shepherd model.

    Parameters:
    probabilities (array-like): Probabilities of each choice, should sum to 1.

    Returns:
    rewards (np.ndarray): Estimated rewards, centered to remove the constant offset.
    """
    probabilities = np.array(probabilities)

    if not np.isclose(probabilities.sum(), 1):
        raise ValueError("Probabilities must sum to 1.")

    if np.any(probabilities < 0):
        raise ValueError("Probabilities must be  nonnegative to compute log.")

    # Compute raw rewards, adding a small value to avoid log(0)
    raw_rewards = np.log(probabilities + 1e-15)

    # Center rewards by subtracting the mean
    centered_rewards = raw_rewards - np.mean(raw_rewards)

    return centered_rewards


def estimate_rewards_(questions_probabilities):
    """
    Estimate rewards from given choice probabilities using the Luce-Shepherd model.

    Parameters:
    questions_probabilities (dict): Probabilities of each choice, should sum to 1.

    Returns:
    rewards_dict (dict): Estimated rewards for each option in each question.
    """
    options = list(questions_probabilities.keys())
    probabilities = np.array([questions_probabilities[key] for key in options])
    if sum(probabilities) == 0:
        rewards_q = np.zeros(len(probabilities))
    else:
        rewards_q = estimate_rewards(probabilities)
    rewards_dict = {options[i]: rewards_q[i] for i in range(len(options))}
    return rewards_dict


def figure():
    fig = plt.figure(figsize=(3.2, 1.7))

    return fig


def subplots_adjust():
    plt.subplots_adjust(left=0.19, right=0.84, top=1, bottom=0.235)
