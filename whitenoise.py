import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

# WHITE NOISE
# mean = 0
# std = 1 
# num_samples = 500
# samples = np.random.normal(mean, std, size=num_samples)
# plt.plot(samples)
# plt.show()

# Set the range and step of x manually
start = 0
stop = 5
step = .1

mean = 0
std = .5
num_samples = int(abs(stop - start) / step)
y = np.random.normal(mean, std, size=num_samples)
x = np.arange(start, stop, step)

y_smooth = signal.savgol_filter(y, window_length=10, polyorder=3, mode="nearest")
y_smooth2 = signal.savgol_filter(y_smooth, window_length=15, polyorder=3, mode = "nearest")


# Adjust ends to be the same max value
def adjust_y(arr):
    y_max = max(arr)
    y_adjusted = arr
    y_adjusted[0] = y_max
    y_adjusted[-2] = y_max

    # y_adjusted[1] = abs((y_max + y_adjusted[2])//2)
    # y_adjusted[-3] = abs((y_max + y_adjusted[-4])//2)

    # All points below 0
    for i in range(len(y_adjusted)):
        y_adjusted[i] = y_adjusted[i] - y_max

    return y_adjusted


def estimate_curve():
    matrix = []
    for i in range(num_samples-1):
        row = [1]
        for j in range(num_samples//2-1):
            row.append(np.sin((j+1)*x[i]))
            row.append(np.cos((j+1)*x[i]))
        matrix.append(row)

    # Logging for shape
    # print(matrix)
    # print("Number of rows", len(matrix))
    # print("Number of columns", len(matrix[0]))
    # print("y smooth twice dimension:", len(y_smooth2[0:num_samples-1]))
    # print("Length of x:", len(x))

    y_adjusted = adjust_y(y_smooth2)
    c = np.linalg.solve(matrix, y_adjusted[:-1])

    # Print the function with constants
    strings = ["sin", "cos"]
    equation = [str(c[0]), "+"]
    for i in range(1, num_samples-1):
        equation.append(str(c[i]))
        equation.append(strings[(i+1)%2])
        equation.append(str((i+1)//2))
        equation.append("x+")
    equation.pop()
    equation.append("x")
    print("".join(equation))    

    # Find the y values from the estimated curve
    a = np.array(matrix)
    est_y = a.dot(c)
   
    return est_y[:len(x)-1]



plt.figure(figsize=(12, 4))
plt.plot(x, y, label = "original")
plt.plot(x, y_smooth, label="y_smoothed_once")
plt.plot(x, y_smooth2, linewidth=3, label="y_smoothed_twice")
plt.plot(x, adjust_y(y_smooth2), linewidth=4, label="y_adjusted")
plt.plot(x[:len(x)-1], estimate_curve(), label="estimated function")
plt.legend()
plt.show()
