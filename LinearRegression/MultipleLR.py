import numpy as np
import pandas as pd

class MLR:
    def __init__(self):
        self.betas=None

    def fit(self,x,y):
        self.n=len(x)
        x=np.c_[np.ones((self.n,1)),x]
        self.betas=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        self.y_mean=np.mean(y)
        self.y_hat=x.dot(self.betas)


    def predict(self,x):
        x=np.c_[np.ones((self.n,1),x)]
        return x.dot(self.betas)

df=pd.read_csv('real_data2.csv')
X = df[["size", "year"]]
y = df["price"]
model = MLR()
model.fit(X,y)
