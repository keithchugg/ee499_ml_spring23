import numpy as np
import matplotlib.pyplot as plt

#load data
x = np.genfromtxt('house-votes-84.csv', delimiter=',', skip_header=True)

#You will write some parts related to clustering. You need to complete 5 Tasks.
#This is a step by step guide to run kmeans algorithm from scratch
def kmeans_fit(x, N_clusters, N_iterations, num_init, epsilon=1e-3):
    P, N = x.shape
    distances = np.zeros((P, N_clusters))
    loss_best = 1e10
    cluster_heads_best = np.zeros((N_clusters, N))
    cluster_assignments = np.zeros(P, dtype=int)
    i = 1
    
    for n in range(num_init):

        ### TASK 1: sample "cluster_heads" from a uniform distribution [-1,1]. Make sure the size of "cluster_heads" is specified correctly
        #   ADD YOUR CODE HERE
        ###END OF TASK 1

        not_done = True
        last_loss = 1e10
        i = 1

        while not_done:
            
            ### TASK 2: Calculate the Euclidean distance "distances" from every point in the data to every "cluster_heads".
            #   ADD YOUR CODE HERE
            ### END OF TASK 2

            ### TASK 3: "cluster_assignments" calculation. Hint: How do you assign points to clusters?
            #   ADD YOUR CODE HERE
            ### END OF TASK 3

            ### TASK 4: compute the loss. You need to loop over N_clusters and evaluate the total distance of points in a cluster
            loss = 0.0
            #   ADD YOUR CODE HERE
            loss = loss / P
            ### END OF TASK 4

            ### check if we are done
            loss_change_fractional = (last_loss - loss) / loss

            # print(f'iteration = {i},  loss = {loss : 3.2e} last_loss = {last_loss : 3.2e} frac loss-delta: {loss_change_fractional : 3.2e}')
            if loss_change_fractional < epsilon or i == num_init:
                not_done = False
                # print(f'this initialiation done: iteration = {i}, fractional loss = {loss_change_fractional : 3.2e}')
            else:
                i += 1
                last_loss = loss
                ### TASK 5: compute new "cluster_heads". You need to loop over N_clusters and evaluate the new centroid location
                #   ADD YOUR CODE HERE
                ### END OF TASK 5
                
        if loss < loss_best:
            cluster_heads_best = cluster_heads
            cluster_assignments_best = cluster_assignments
            loss_best = loss
            # print(f'n = {n}, new best loss: {loss_best}')

    return cluster_heads_best, cluster_assignments_best, loss_best 

N_clusters = np.arange(1, 5)
loss_values = []
cluster_heads = []
cluster_assignments = []

for num_clusters in N_clusters:
    print(f'\nNumber of Clusters = {num_clusters}')
    heads, assigments, loss = kmeans_fit(x, num_clusters, 100, 5)
    loss_values.append(loss)
    cluster_heads.append(heads)
    cluster_assignments.append(assigments)

    plt.figure()
    for m in range(num_clusters):
        plt.plot(np.arange(1,17), np.mean(x[assigments == m], axis = 0), linestyle='--', marker='o', label=f'votes cluster {m}')
    plt.grid(':')
    plt.legend()
    plt.xlabel('initiative')
    plt.ylabel('average vote for cluster')
    plt.show()

plt.figure()
plt.plot(N_clusters, loss_values, color='b', linestyle='--', marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('K-means Loss per Data Point')
plt.grid(':')
plt.show()