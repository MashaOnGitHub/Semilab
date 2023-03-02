import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import leastsq
from scipy import linalg

# WHITE NOISE
# mean = 0
# std = 1 
# num_samples = 200
# samples = np.random.normal(mean, std, size=num_samples)
# plt.plot(samples)
# plt.show()

# Set the range and step of x manually
start = 0
stop = 6
step = 0.02

mean = 0
std = .5
num_samples = int(abs(stop - start) / step)
samples = np.random.normal(mean, std, size=num_samples)
x = np.arange(start, stop, step)
# y = np.array(x**2+2*np.sin(x*np.pi)) 
# y = y + np.array(np.random.random(len(x))*2.3)
# x = np.linspace(0,2*np.pi,100)
# y = np.sin(x) + np.random.random(100) * 0.8
y = samples

y_smooth = signal.savgol_filter(y, window_length=10, polyorder=3, mode="nearest")
y_smooth2 = signal.savgol_filter(y_smooth, window_length=10, polyorder=3, mode = "nearest")
print(len(y_smooth2))


def estimateCurve():
    matrix = []
    # row = [1]
    for i in range(7): # len(y_smooth2)
        # row.append(np.sin((i+1)*x[i]))
        # row.append(*np.cos((i+1)*x[i]))
        row = [1, np.sin(x[i]), np.cos(x[i]), np.sin(2*x[i]), np.cos(2*x[i]), np.sin(3*x[i]), np.cos(3*x[i])]
        matrix.append(row)

    c = np.linalg.solve(matrix, y_smooth2[0:7])

    print(str(c[0]) + " + " + str(c[1]) + " * sin(x)" + str(c[2]) + " * cos(x)" + str(c[3]) + " * sin(2x)" + str(c[4]) + " * cos(2x)" + str(c[5]) + " * sin(3x)" + str(c[6]) + " * cos(3x)")
    est_y = []
    for i in range(len(x)):
        est_y.append(c[0] + c[1]*np.sin(x[i]) + c[2]*np.cos(x[i]) + c[3]*np.sin(2*x[i]) + c[4]*np.cos(2*x[i]) + c[5]*np.sin(3*x[i]) + c[6]*np.cos(3*x[i]))

    return est_y[0:len(x)-2]


    

plt.figure(figsize=(12, 4))
plt.plot(x, y, label = "original")
plt.plot(x, y_smooth, label="y_smoothed_once")
plt.plot(x, y_smooth2, linewidth=3, label="y_smoothed_twice")
plt.legend()
# plt.plot(x[0:len(x)-2], estimateCurve(), label="estimated function")
plt.show()

