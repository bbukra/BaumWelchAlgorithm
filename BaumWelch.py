import numpy as np

def _do_Forward_Backward_Pass(initial_State_Distributions, transition_Mat, state_Output_Distributions, _U):
    corpus_Length = len(_U)
    number_Of_States = len(initial_State_Distributions)
    alphas = np.array([[0] * number_Of_States] * corpus_Length)
    betas  = np.array([[0] * number_Of_States] * corpus_Length)
    rescaling_Coeffs = np.array([0] * corpus_Length)

    alphas[:, 0] = np.matmul(np.transpose(transition_Mat[_U[0]]), state_Output_Distributions)
    rescaling_Coeffs[0] = 1 / np.sum(alphas[:, 0])
    for t in range(1, corpus_Length):
        alphas[:, t] = np.matmul((initial_State_Distributions * alphas[:,t-1]), np.transpose(B[U[t],:]))
        rescaling_Coeffs[t] = 1 / np.sum(alphas[:, t])
        alphas[:, t] = rescaling_Coeffs[t] * alphas[:, t]

    betas[:, corpus_Length] = np.array([1] * number_Of_States)
    betas[:, corpus_Length] = rescaling_Coeffs[corpus_Length] * betas[:, corpus_Length]

    for t in reversed(range(0, corpus_Length - 1)):
        betas[:, t] = \
            np.matmul(
                np.matmul(
                    np.transpose(initial_State_Distributions),
                    np.transpose(transition_Mat[_U[t + 1], :])),
                betas[:, t + 1])
        betas[:, t] = rescaling_Coeffs[t] * betas[:, t]

    return alphas, betas

def _calc_Gammas_Xis(initial_State_Distributions, transition_Mat, state_Output_Distributions):
    alphas, betas = _do_Forward_Backward_Pass(initial_State_Distributions, transition_Mat, state_Output_Distributions)


def _perform_Estep():
    gammas, xis = _calc_Gammas_Xis()

def _perform_Mstep():
    pass

"""
Input: L sequences of observations, number of states in HMM
Output: Estimation for HMM parameters
"""

tol = 0.005
def BaumWelchAlgorithm(observations, num_Of_States = 2):
    initial_State_Distributions = [1/num_Of_States] * num_Of_States
    transition_Mat = [[1/num_Of_States] * num_Of_States] * num_Of_States
    state_Output_Distributions = [1/num_Of_States] * num_Of_States
    _perform_Estep()
    _perform_Mstep()



