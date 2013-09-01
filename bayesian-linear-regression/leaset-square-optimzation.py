import numpy as np
from scipy.optimize import leastsq
import pylab as plt

N = 1000 # number of data points
t = np.linspace(0, 4*np.pi, N)
data = 3.0*np.log(t+0.001) + 0.5 + np.random.randn(N) # create artificial data with noise

guess_a = np.mean(data)
guess_b = 3*np.std(data)/(2**0.5)
guess_c = 0

# we'll use this to plot our first estimate. This might already be good enough for you
data_first_guess = guess_b*np.log(t+guess_c) + guess_a 

# Define the function to optimize, in this case, we want to minimize the difference
# between the actual data and our "guessed" parameters
optimize_func = lambda x: x[0]*np.log(t+x[2]) + x[1] - data
est_a, est_b, est_c = leastsq(optimize_func, [guess_a, guess_b, guess_c])[0]

# recreate the fitted curve using the optimized parameters
data_fit = est_a*np.log(t+est_c) + est_b

plt.plot(data, '.')
plt.plot(data_fit, label='after fitting')
plt.plot(data_first_guess, label='first guess')
plt.legend()
plt.show()
