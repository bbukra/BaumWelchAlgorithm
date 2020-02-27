import numpy as np

def _do_Forward_Backward_Pass(transition_Mat, state_Output_Distributions, initial_State_Distributions,  _U):
    corpus_Length = len(_U)
    number_Of_States = len(transition_Mat)

    alphas = np.zeros((number_Of_States, corpus_Length), dtype=np.float)
    betas = np.zeros((number_Of_States, corpus_Length), dtype=np.float)
    rescaling_Coeffs = np.array([0] * corpus_Length)

    alphas[:, 0] = np.multiply(np.transpose(state_Output_Distributions[_U[0] - 1]), initial_State_Distributions)
    rescaling_Coeffs[0] = 1 / np.sum(alphas[:, 0])
    alphas[:, 0] = rescaling_Coeffs[0] * alphas[:, 0]
    for t in range(1, corpus_Length):
        alphas[:, t] = \
            np.multiply(
                np.matmul(transition_Mat, alphas[:,t-1]),
                np.transpose(state_Output_Distributions[(_U[t] - 1), :]))
        rescaling_Coeffs[t] = 1 / np.sum(alphas[:, t])
        alphas[:, t] = rescaling_Coeffs[t] * alphas[:, t]

    betas[:, corpus_Length - 1] = np.array([1] * number_Of_States)
    betas[:, corpus_Length - 1] = rescaling_Coeffs[corpus_Length - 1] * betas[:, corpus_Length - 1]

    for t in reversed(range(0, corpus_Length - 1)):
        betas[:, t] = \
            np.multiply(
                np.matmul(
                    np.transpose(transition_Mat),
                    np.transpose(state_Output_Distributions[_U[t + 1] - 1, :])),
                betas[:, t + 1])
        betas[:, t] = rescaling_Coeffs[t] * betas[:, t]
    return alphas, betas

def _calc_Gammas_Xis(transition_Mat, state_Output_Distributions, initial_State_Distributions, observations):
    alphas, betas = _do_Forward_Backward_Pass(transition_Mat, state_Output_Distributions, initial_State_Distributions, observations)
    gammas = np.zeros((number_Of_States, corpus_Length), dtype=np.float)
    xis = np.zeros((number_Of_States, number_Of_States, corpus_Length), dtype=np.float)

    for t in range(len(alphas[0])):
        sum_Over_J_atj_btj = 0
        for j in range(len(alphas)):
            sum_Over_J_atj_btj = sum_Over_J_atj_btj + (alphas[j , t] * betas[j, t])
        for i in range(len(alphas)):
            # γt(i) = αt(i)*βt(i) / sum_j (αt(j)*βt(j))
            gammas[i, t] = (alphas[i, t] * betas[i, t]) / sum_Over_J_atj_btj
            for j in range(len(alphas)):
                prob_O_tplus1_Given_s_tplus1_j = state_Output_Distributions[j, observations[t + 1] - 1]
                prob_s_tplus1_j_Given_s_t_i = transition_Mat[i, j]
                xis[i, j, t] = (alphas[i, t] * betas[j, t + 1] * prob_s_tplus1_j_Given_s_t_i * prob_O_tplus1_Given_s_tplus1_j) / sum_Over_J_atj_btj
    return gammas, xis

def _perform_Estep(transition_Mat, state_Output_Distributions, initial_State_Distributions, observations):
    gammas, xis = _calc_Gammas_Xis(transition_Mat, state_Output_Distributions, initial_State_Distributions, observations)

def _perform_Mstep(transition_Mat, state_Output_Distributions, initial_State_Distributions, observations):
    pass

"""
Input: A single sequence of observations;
       The number of states in the HMM;
Output: Estimation for HMM parameters
"""

tol = 0.005
def BaumWelchAlgorithm(observations, num_Of_States = 3):
    m = len(observations)
    observations = np.array(observations)
    initial_State_Distributions = np.array([1/num_Of_States] * num_Of_States)
    transition_Mat = np.array([[1/num_Of_States] * num_Of_States] * num_Of_States)
    state_Output_Distributions = np.array([[1/m] * num_Of_States] * m)
    _perform_Estep(transition_Mat, state_Output_Distributions, initial_State_Distributions, observations)
    _perform_Mstep(transition_Mat, state_Output_Distributions, initial_State_Distributions, observations)

