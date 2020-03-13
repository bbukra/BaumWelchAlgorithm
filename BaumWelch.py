import numpy as np
from copy import deepcopy
"""
    this is an implementation of the baum welch algorithm as presented in:
    https://www.cs.bgu.ac.il/~inabd171/wiki.files/lecture20_handouts.pdf
    a lecture slide by prof. aryeh kontorovich and sivan sabato from bgu
"""

def _do_forward_backward_pass(transition_mat, state_output_distributions, initial_state_distributions,  observations_seq):
    corpus_length = len(observations_seq)
    number_of_states = len(transition_mat)

    alphas = np.zeros((number_of_states, corpus_length), dtype=np.float)
    betas = np.zeros((number_of_states, corpus_length), dtype=np.float)
    rescaling_coeffs = np.array([0] * corpus_length)

    alphas[:, 0] = np.multiply(np.transpose(state_output_distributions[int(observations_seq[0] - 1)]), initial_state_distributions)
    rescaling_coeffs[0] = 1 / np.sum(alphas[:, 0])
    alphas[:, 0] = rescaling_coeffs[0] * alphas[:, 0]
    for t in range(1, corpus_length):
        alphas[:, t] = \
            np.multiply(
                np.matmul(transition_mat, alphas[:,t-1]),
                np.transpose(state_output_distributions[int(observations_seq[t] - 1), :]))
        rescaling_coeffs[t] = 1 / np.sum(alphas[:, t])
        alphas[:, t] = rescaling_coeffs[t] * alphas[:, t]

    betas[:, corpus_length - 1] = np.array([1] * number_of_states)
    betas[:, corpus_length - 1] = rescaling_coeffs[corpus_length - 1] * betas[:, corpus_length - 1]

    for t in reversed(range(0, corpus_length - 1)):
        betas[:, t] = \
            np.multiply(
                np.matmul(
                    np.transpose(transition_mat),
                    np.transpose(state_output_distributions[int(observations_seq[t + 1] - 1), :])),
                betas[:, t + 1])
        betas[:, t] = rescaling_coeffs[t] * betas[:, t]
    return alphas, betas

def _calc_gammas_xis(transition_mat, state_output_distributions, initial_state_distributions, observations):
    number_of_states = len(initial_state_distributions)
    corpus_length = len(observations)
    alphas, betas = \
        _do_forward_backward_pass(transition_mat, state_output_distributions, initial_state_distributions, observations)
    gammas = np.zeros((number_of_states, corpus_length), dtype=np.float)
    xis = np.zeros((number_of_states, number_of_states, corpus_length - 1), dtype=np.float)

    for t in range(len(alphas[0])):
        sum_over_j_atj_btj = 0
        for j in range(len(alphas)):
            sum_over_j_atj_btj = sum_over_j_atj_btj + (alphas[j , t] * betas[j, t])
        for i in range(len(alphas)):
            gammas[i, t] = (alphas[i, t] * betas[i, t]) / sum_over_j_atj_btj
            if (t < len(alphas[0]) - 1):
                for j in range(len(alphas)):
                    prob_o_tplus1_given_s_tplus1_j = state_output_distributions[int(observations[t + 1] - 1), j]
                    prob_s_tplus1_j_given_s_t_i = transition_mat[i, j]
                    xis[i, j, t] = (alphas[i, t] * betas[j, t + 1]
                                    * prob_s_tplus1_j_given_s_t_i
                                    * prob_o_tplus1_given_s_tplus1_j) / sum_over_j_atj_btj
    return gammas, xis

def _perform_estep(transition_mat, state_output_distributions, initial_state_distributions, observations):
    gammas = [0] * (len(observations)) # initialize an array of size l
    xis = [0] * (len(observations))    # initialize an array of size l
    for i in range(len(observations)):
        # each entry of these arrays is an array of size n_l (sequence length)
        gammas[i], xis[i] = \
            _calc_gammas_xis(transition_mat, state_output_distributions, initial_state_distributions, observations[i])
    return gammas, xis

global prev_transition_probs_mle
global prev_output_probs_mle
global transition_probs_mle
global output_probs_mle

def _update_initial_state_distribution(initial_state_distributions, gammas, l):
    sum_gammas_l_i = 0
    for i in range(len(initial_state_distributions)):
        for l in range(l):
            sum_gammas_l_i = sum_gammas_l_i + gammas[l][i][0]
        initial_state_distributions[i] = (1 / l) * sum_gammas_l_i
        sum_gammas_l_i = 0
    return initial_state_distributions

def _update_transition_mat(transition_mat, observations, xis, l, number_of_states):
    sum_xis_l_i_j_t = 0
    for i in range(number_of_states):
        for j in range(number_of_states):
            for l in range(l):
                for t in range(len(observations[l]) - 1):
                    sum_xis_l_i_j_t = sum_xis_l_i_j_t + xis[l][i][j][t]
            transition_mat[i, j] = sum_xis_l_i_j_t
            sum_xis_l_i_j_t = 0
    return transition_mat

def _update_transition_probs_mle(transition_mat, number_of_states):
    global prev_transition_probs_mle
    global transition_probs_mle

    prev_transition_probs_mle = deepcopy(transition_probs_mle)
    for i in range(number_of_states):
        row_sum = np.sum(transition_mat[:, i])
        for j in range(number_of_states):
            transition_probs_mle[i][j] = transition_mat[i, j] / row_sum

def _update_state_output_distributions(state_output_distributions, observations, gammas, l, number_of_states, m):
    sum_gammas_t_l_i_where_k_is_observed_at_index_t_in_observation_seq = 0
    for i in range(number_of_states):
        for k in range(1, m + 1):
            for l in range(l):
                for t in range(len(observations[l])):
                    indicator_index_t_in_observation_seq_l_is_k = 1 if observations[l][t] == k else 0
                    sum_gammas_t_l_i_where_k_is_observed_at_index_t_in_observation_seq = \
                        sum_gammas_t_l_i_where_k_is_observed_at_index_t_in_observation_seq \
                        + (gammas[l][i][t]) * indicator_index_t_in_observation_seq_l_is_k

            state_output_distributions[k - 1, i] = sum_gammas_t_l_i_where_k_is_observed_at_index_t_in_observation_seq
            sum_gammas_t_l_i_where_k_is_observed_at_index_t_in_observation_seq = 0
    return state_output_distributions

def _update_output_probs_mle(state_output_distributions, number_of_states, m):
    global prev_output_probs_mle
    global output_probs_mle

    prev_output_probs_mle = deepcopy(output_probs_mle)
    for i in range(number_of_states):
        row_sum = np.sum(state_output_distributions[:, i])
        for k in range(m):
            output_probs_mle[k][i] = state_output_distributions[k, i] / row_sum

def _perform_mstep(transition_mat, state_output_distributions, initial_state_distributions, observations, gammas, xis, m):
    number_of_states = len(initial_state_distributions)
    l = len(observations)

    initial_state_distributions = _update_initial_state_distribution(initial_state_distributions, gammas, l)

    transition_mat = _update_transition_mat(transition_mat, observations, xis, l, number_of_states)

    _update_transition_probs_mle(transition_mat, number_of_states)

    state_output_distributions = \
        _update_state_output_distributions(state_output_distributions, observations, gammas, l, number_of_states, m)

    _update_output_probs_mle(state_output_distributions, number_of_states, m)

    return initial_state_distributions, transition_mat, state_output_distributions

def _calc_transition_mle_difference(number_of_states):
    global transition_probs_mle
    global prev_transition_probs_mle
    transition_mle_difference = 0

    for i in range(number_of_states):
        for j in range(number_of_states):
            transition_mle_difference = transition_mle_difference \
                                        + abs(transition_probs_mle[i, j] - prev_transition_probs_mle[i, j])
    return transition_mle_difference

def _calc_output_mle_difference(number_of_states, m):
    global output_probs_mle
    global prev_output_probs_mle
    output_mle_difference = 0

    for i in range(number_of_states):
        for j in range(m):
            output_mle_difference = output_mle_difference \
                                    + abs(output_probs_mle[j, i] - prev_output_probs_mle[j, i])
    return output_mle_difference

def _converged(number_of_states, m):
    tolerance = 0.005
    transition_mle_difference = _calc_transition_mle_difference(number_of_states)
    output_mle_difference = _calc_output_mle_difference(number_of_states, m)

    if (transition_mle_difference < tolerance
            and output_mle_difference < tolerance):
        return true
    else:
        return false


"""
input: l sequences of observations;
       m - the maximum value of an observation
       the number of states in the hmm;
output: estimation for hmm parameters
"""

def baumwelchalgorithm(observations, m, num_of_states = 3):
    global transition_probs_mle
    global output_probs_mle
    l = len(observations)

    # data initialization:
    observations = np.array(observations)
    for i in range(l):
        observations[i] = np.array(observations[i], dtype=np.float)

    transition_mat = np.array([[1/num_of_states] * num_of_states] * num_of_states, dtype=np.float)
    initial_state_distributions = np.array([1/num_of_states] * num_of_states, dtype=np.float)
    state_output_distributions  = np.array([[1/m] * num_of_states] * m, dtype=np.float)

    transition_probs_mle = np.array([[0] * num_of_states] * num_of_states, dtype=np.float)
    output_probs_mle     = np.array([[0] * num_of_states] * m, dtype=np.float)


    while(true):
        gammas, xis = \
            _perform_estep(transition_mat, state_output_distributions, initial_state_distributions, observations)
        initial_state_distributions, transition_mat, state_output_distributions = \
            _perform_mstep(transition_mat, state_output_distributions, initial_state_distributions, observations, gammas, xis, m)
        print("initial_state_distributions   :\n", initial_state_distributions   , "\n\n",
              "transition_mat                :\n", transition_mat                , "\n\n",
              "state_output_distributions    :\n", state_output_distributions    , "\n\n\n\n",
              "transition_probs_mle          :\n", transition_probs_mle          , "\n\n",
              "prev_transition_probs_mle     :\n", prev_transition_probs_mle     , "\n\n",
              "output_probs_mle              :\n", output_probs_mle              , "\n\n",
              "prev_output_probs_mle         :\n", prev_output_probs_mle         )
        if(_converged(num_of_states, m)):
            break

    return initial_state_distributions, transition_mat, state_output_distributions

baumwelchalgorithm([[2, 2, 3, 4]], 4)
