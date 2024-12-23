# Régression Linéaire Simple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean

# Importer le dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Construction du modèle
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

regressor.predict([[15]])  

#ici on va voir les valeurs des parametres de notre equation
# y=ax+b
regressor.coef_
regressor.intercept_

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title(u'Salaire vs Expérience')
plt.xlabel(u'Expérience')
plt.ylabel(u'Salaire')
plt.show()

# evaluer notre modele 
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
mse
rmse = np.sqrt(mse)
rmse




