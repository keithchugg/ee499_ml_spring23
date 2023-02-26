
import numpy as np

X = np.random.normal(0,1, (28,28))

from scipy.signal import convolve2d
# X.shape is (H, W), for MNIST/FMNIST, this is (28, 28)
H_HL = np.array([[1, -1], [1, -1]])     ## define 2x2 impulse response
Y = convolve2d(X, H_HL)                 ## perform 2D convolution, produces a 29 x 29 image because of edge effects
X_HL = Y[1::2].T[1::2]                  ## downsamples and shifts to get the desired 14 x 14 image