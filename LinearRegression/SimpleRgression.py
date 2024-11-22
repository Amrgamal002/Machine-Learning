import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SLR:
    def __init__(self):
        self.B0 = 0
        self.B1 = 0

    def fit(self, x, y):
        self.x = x
        self.y = y
        first_sum = np.sum(self.x * (self.y - np.mean(self.y)))
        second_sum = np.sum(self.x * (self.x - np.mean(self.x)))

        self.B1 = first_sum / second_sum
        self.B0 = np.mean(self.y) - self.B1 * np.mean(self.x)
        self.y_hat = self.B0 + self.B1 * self.x

    def predict(self, x):
        return self.B0 + self.B1 * x

    def r_square(self):
        sst = np.sum((self.y - np.mean(self.y)) ** 2)
        sse = np.sum((self.y - self.y_hat) ** 2)
        return 1 - sse / sst

x = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y = np.array([250, 300, 480, 430, 630, 730])

mod = SLR()
mod.fit(x, y)
yhat = mod.y_hat
print(yhat)
plt.scatter(x, y)
plt.plot(x, yhat)
plt.show()
print(mod.predict(5))
print(mod.r_square())