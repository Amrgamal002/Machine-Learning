import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self,n_itreation=1000,learning_rate=.01):
        self.alpha=learning_rate
        self.n_itreation=n_itreation
        self.b=None
        self.w=None

    def fit(self,x,y):
        n,k=x.shape
        self.b=0
        self.w=np.zeros((n,k))
        for _ in range(self.n_itreation):
            self.y_hat=np.dot(x,self.w)+self.b
            dw=2/n*np.dot(x,(self.y_hat-y))
            db=2/n*np.sum(self.y_hat-y)
            self.b=self.b-self.alpha*db
            self.w=self.w-self.alpha*dw

    def predict(self,x):
        return np.dot(x,self.w)+self.b



