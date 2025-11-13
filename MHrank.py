# Metropolis-Hastings MCMC algorithm for sampling skills in the probit rank model
# -gtc 20/09/2025
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

def MH_sample(games, num_players, num_its, proposal_std=np.sqrt(0.5), prior_std=1.0, return_acceptance=False):

    # pre-process data:
    # array of games for each player, X[i] = [(other_player, outcome), ...]
    X = [[] for _ in range(num_players)] 
    for a, (i,j) in enumerate(games):
        X[i].append((j, +1))  # player i beat player j
        X[j].append((i, -1))  # player j lost to payer i
    for i in range(num_players):
        X[i] = np.array(X[i])

    # array that will contain skill samples
    skill_samples = np.zeros((num_players, num_its))

    w = np.zeros(num_players)  # skill for each player
    # track how many proposals are accepted for each player
    accept_counts = np.zeros(num_players, dtype=int)

    for itr in tqdm(range(num_its)):
        for i in range(num_players):
            j, outcome = X[i].T

            # current local log-prob (use prior with specified std)
            lp1 = norm.logpdf(w[i], loc=0, scale=prior_std) + np.sum(norm.logcdf(outcome*(w[i]-w[j])))

            # proposed new skill and log-prob (proposal_std is the proposal std)
            w_proposed = w[i] + np.random.normal(0, proposal_std)
            lp2 = norm.logpdf(w_proposed, loc=0, scale=prior_std) + np.sum(norm.logcdf(outcome*(w_proposed-w[j])))

            # accept or reject move:
            if np.log(np.random.uniform()) < lp2 - lp1:
                w[i] = w_proposed
                accept_counts[i] += 1

        skill_samples[:, itr] = w

    if return_acceptance:
        # acceptance rate per player = accepted proposals / number of iterations
        acceptance_rates = accept_counts.astype(float) / float(num_its)
        return skill_samples, acceptance_rates
    return skill_samples

