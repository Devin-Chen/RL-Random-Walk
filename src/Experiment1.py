import numpy as np
from RandomWalk import RandomWalk
import math
import matplotlib.pyplot as plt
from datetime import datetime

def run_exp1():
    lams = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rsmes = []
    training_size = 100
    sequence_size = 10
    alpha = 0.01
    epsilon = 0.001
    random_walk = RandomWalk(training_size, sequence_size)
    random_walk.generate_data()
    filename = 'generate_fig3_a' + str(alpha) +'_e' + str(epsilon) + '_' +\
               datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt'
    f = open('results/' + filename, 'w', )
    f.write('lambda\tRSME\tSE\n')
    for lam in lams:
        weights, rsme, se = get_predictions(lam, random_walk, training_size, sequence_size, alpha, epsilon)
        rsmes.append(rsme)
        f.write('{}\t{}\t{}\n'.format(lam, rsme, se))
        print 'at lambda = ', lam, ' --> rsme = ', rsme, ' | weights = ', weights
    f.close()
    plt.plot(lams, rsmes)
    plt.xlabel('Lambda')
    plt.ylabel('Error (RSME)')
    plt.title('Random Walk - Reproducing Figure 3')
    plt.grid(True)
    plt.show()


def get_predictions(lam, random_walk, training_size, sequence_size, alpha, epsilon):
    # perform experiment on random walk to replicate figure 3 results
    rsme_list = []
    for i in range(training_size):
        weights = [0.5, 0.5, 0.5, 0.5, 0.5]
        observations = random_walk.observations[i]
        converged = False
        # iter = 0
        # stop when weight converges
        while not converged:
            delta_w = 0
            for j in range(sequence_size):
                obs = observations[j]
                for t in range(1, len(obs) + 1):
                    ind = t - 1
                    p_t = np.dot(weights, obs[ind])
                    p_tn1 = get_p_tn1(obs, ind, weights)
                    discount_delta_w = 0
                    for k in range(1, t + 1):
                        temp = np.multiply(lam ** (t - k), obs[k - 1])
                        discount_delta_w = np.add(temp, discount_delta_w)
                    dw = np.multiply(alpha * (p_tn1 - p_t), discount_delta_w)
                    delta_w += dw
            converged = np.all(np.abs(delta_w) < epsilon)
            weights += delta_w
            # print i, ' | iter: ', iter, ' --> weights: ', weights, 'delta: ', delta_w
            # iter += 1
        err = compute_error(weights)
        rsme_list.append(err)
    stdev = np.std(rsme_list, ddof=1)
    se = stdev / math.sqrt(len(rsme_list))
    return weights, np.mean(rsme_list), se

def compute_error(weights):
    expected = np.array([1.0/6, 1.0/3, 1.0/2, 2.0/3, 5.0/6])
    rsme = math.sqrt(np.mean(np.power(np.subtract(weights, expected), 2)))
    return rsme


def get_p_tn1(obs, ind, weights):
    if ind == len(obs) - 1:
        # at last observation of sequence
        if obs[ind] == [0,0,0,0,1]:
            return 1
        elif obs[ind] == [1,0,0,0,0]:
            return 0
    else:
        # not last observation of sequence
        return np.dot(weights, obs[ind + 1])





if __name__ == '__main__':
    run_exp1()