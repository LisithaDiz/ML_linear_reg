import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data.csv')


def loss_func(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].x  # x attribute from the i-th row of a pandas DataFrame points using the iloc indexer.
        y = points.iloc[i].y  # y attribute from the i-th row of a pandas DataFrame points using the iloc indexer.
        total_error += (y - m * x + b) ** 2
    return total_error / float(len(points))


def gradient_decent(m_now, b_now, points, L):
    m_grad = 0
    b_grad = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        m_grad += -2 / n * x * (y - (m_now * x + b_now))
        b_grad += -2 / n * (y - (m_now * x + b_now))

        m = m_now - m_grad * L
        b = b_now - m_grad * L
        return m, b


m = 0
b = 0
L = 0.0001  # learning rate
metrix_x = np.zeros(20000)
metrix_y = np.zeros(20000)
epochs = 20000  # 100000   # number of interations

for i in range(epochs):
    m, b = gradient_decent(m, b, data, L)
    if i % 1000 == 0:
        print(i)
    # array need to draw the epochs vs loss
    metrix_x[i] = i
    metrix_y[i] = loss_func(m, b, data)

print(m, b)
print(loss_func(m, b, data))

plt.subplot(1, 2, 1)
plt.scatter(data.x, data.y, color="red")
plt.plot(list(range(0, 20)), [m * x + b for x in range(0, 20)], color='black')

plt.subplot(1, 2, 2)
plt.plot(metrix_x, metrix_y)
plt.xlabel('n th epoch')
plt.ylabel('Loss value')
plt.show()
