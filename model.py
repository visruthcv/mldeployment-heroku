import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

dataset = pd.read_excel('dataset.xlsx')

X = dataset.iloc[:, :4]

y = dataset.iloc[:, -1]

regress = LinearRegression()

regress.fit(X, y)

pickle.dump(regress, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
