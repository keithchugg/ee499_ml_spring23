import numpy as np
import matplotlib.pyplot as plt

### HW 1 problem 1
def M_func(P, N):
    return P ** N

Ps = np.asarray([2, 4, 16, 32, 64])
plt.figure()
plt.semilogy(Ps, M_func(Ps, 10), marker='o', color='b', label='total number of points')
plt.grid(':')
plt.xlabel('P')
plt.ylabel('Total number of points to evaluate g(.)')
plt.legend()

Ns = np.asarray([2, 4, 16])
plt.figure()
plt.semilogy(Ns, M_func(10, Ns), marker='o', color='b', label='total number of points')
plt.grid(':')
plt.xlabel('N')
plt.ylabel('Total number of points to evaluate g(.)')
plt.legend()

### HW 1 problem 2.1 from Watt

grid_1 = np.linspace(-1, 1, 100)

# N = 1
w = grid_1
index_star = np.argmin(w ** 2)
w_star = w[index_star]
g_star = w[index_star] ** 2
print(f'w_star = {w_star}')
print(f'g_star = {g_star}')

# N = 2
grid_1
g_star = 1e9
for w1 in grid_1:
    v1 = w1 ** 2
    for w2 in grid_1:
        v2 = w2 ** 2
        g = v1 + v2
        if g < g_star:
            g_star = g
            w_star = np.asarray([w1, w2])
