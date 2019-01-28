import numpy as np
from RandomWalk import RandomWalk
import math
import matplotlib.pyplot as plt
from datetime import datetime

def run_exp3():
    # params is lamda and best alpha pair
    params = []
    for i in range(11):
        if i < 5:
            params.append([i*0.1, 0.2])
        elif i < 8:
            params.append([i*0.1, 0.15])
        elif i < 10:
            params.append([i*0.1, 0.10])
        else:
            params.append([i*0.1, 0.05])
    rsmes = []
    training_size = 100
    sequence_size = 10
    random_walk = RandomWalk(training_size, sequence_size)
    random_walk.generate_data()
    filename = 'generate_fig5_' +\
               datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt'
    f = open('results/' + filename, 'w', )
    f.write('lambda\tRSME\talpha\tSE\n')
    for i in params:
        weights, rsme, se = get_predictions(i[0], random_walk, training_size, sequence_size, i[1])
        rsmes.append(rsme)
        f.write('{}\t{}\t{}\t{}\n'.format(i[0], rsme, i[1], se))
        print 'at alpha = ', i[1], 'at lambda = ', i[0], ' --> rsme = ', rsme, ' | weights = ', weights
    f.close()
    # plot different lambdas
    lams = [i[0] for i in params]
    plt.plot(lams, rsmes)
    plt.xlabel('Lambda')
    plt.ylabel('Error (RSME)')
    plt.title('Random Walk - Reproducing Figure 5')
    plt.grid(True)
    plt.show()


def get_predictions(lam, random_walk, training_size, sequence_size, alpha):
    # perform experiment on random walk to replicate figure 4 results
    rsme_list = []
    for i in range(training_size):
        weights = [0.5, 0.5, 0.5, 0.5, 0.5]
        observations = random_walk.observations[i]
        for j in range(sequence_size):
            obs = observations[j]
            delta_w = 0
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
    run_exp3()