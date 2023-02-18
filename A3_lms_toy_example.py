import numpy as np 
import matplotlib.pyplot as plt

N = 8
eta = 0.5
N_epochs = 100

w = np.asarray([-2, 4])
x = np.random.choice([-1 , +1], 2 * N).reshape(N, 2)
w_hat = np.zeros(2)

w_hat_hist = np.zeros((N * N_epochs + 1, 2))
w_hat_hist[0] = w_hat[:]
print(f'\nInitial w_hat: {w_hat}\n')
for i in range(N * N_epochs):
    y = np.dot(w, x[i % N])
    y_hat = np.dot(w_hat, x[i % N])
    error = y_hat - y
    update = - eta * error * x[i % N]
    w_hat = w_hat + update
    w_hat_hist[i+1] = w_hat
    print(f'i = {i}\tx_n = {x[i % N]}')
    print(f'i = {i}\ty_n = {y : 0.2f}\ty_hat = {y_hat : 0.2f}\terror = {error : 0.2f}')
    print(f'i = {i}\tupdate[{i}] = {update}')
    print(f'i = {i}\tw_hat[{i}] = {w_hat}\n')


fig, ax = plt.subplots(1, 2, sharex=False, figsize=(12, 4))
iterations = np.arange(N * N_epochs + 1)
ax[0].plot(iterations, w_hat_hist.T[0], color='g', linestyle = '--', marker='o', label=r'$\hat{w}_0$')
ax[0].plot(iterations, w_hat_hist.T[1], color='b', linestyle = '--', marker='o', label=r'$\hat{w}_1$')
ax[0].axhline(w[0], c='g', )
ax[0].axhline(w[1], c='b')
ax[0].text(N * N_epochs / 2, np.mean(w), fr'$\eta = ${float(eta) : 0.2}')
ax[0].set_xlabel('iteration')
ax[0].set_ylabel('w coefficents and estimates')
ax[0].grid(':')
ax[0].legend()
# ax[0].show()
# ax[0].close()

ax[1].plot(w_hat_hist.T[0], w_hat_hist.T[1], color='k', marker='.', label=r'$\hat{\bf w}(i)$')
ax[1].scatter(w[0], w[1], s=100, marker='o', color='r', label=r'$\bf w$')
ax[1].set_xlabel(r'$w_0$')
ax[1].set_ylabel(r'$w_1$')
ax[1].grid(':')
ax[1].legend()
#plt.savefig(f'plots/lms_toy_eta_{eta}.pdf', bbox_inches='tight')
plt.show()
plt.close()


"""
Students should explore:

* what happens if some noise is added to y_n so that it is not a perfect fit?
* what happens if y_n is generated using a linear model with more than D=2?
* is the stability of the LMS dependent on the initial value of w_hat?

"""
