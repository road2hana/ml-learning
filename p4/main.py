import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here

K = np.array([1, 2, 3, 4])

for k in K:
    seeds = np.array([0, 1, 2, 3, 4])
    #k_cost = np.zeros((seeds.shape[0], 2))
    min_cost_seed_i = 0
    mixtures , posts, costs = [], [], []
    for seed_i in range(seeds.shape[0]):
        mixture, post = common.init(X, k, seeds[seed_i])
        mixture, post, cost = kmeans.run(X, mixture, post)
        mixtures.append(mixture)
        posts.append(post)
        costs.append(cost)
        if seed_i > 0 and cost < costs[seed_i-1]:
            min_cost_seed_i = seed_i

    common.plot(X, mixtures[min_cost_seed_i], posts[min_cost_seed_i], "k:"+str(k)+" seed:"+str(min_cost_seed_i))
    print(k, cost, min_cost_seed_i)