# Régression Linéaire Multiple
# Importer les librairies
import numpy as np
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Gérer les variables catégoriques
# on importe la classe pour appliquer la méthode des dummy variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        
         OneHotEncoder(), 
         [3]              
         )
    ],
   remainder='passthrough' 
)

X = transformer.fit_transform(X)

X = X.astype('int')



# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Construction du modèle
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

regressor.predict(np.array([[0,1, 0, 130000, 140000, 300000]]))

#ici on va voir les valeurs des parametres de notre equation
# y=a1x1+a2x2 + a3x3+ a4x4+ a5x5+ b
#pour b c'est regressor.intercept_
regressor.intercept_
# pour a1 a2 a3 a4 a5 c est regressor.coef_
regressor.coef_

from sklearn.metrics import r2_score
#ici on va calculer r2 pour toutes les 6 variables
r2_score(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
mse
rmse = np.sqrt(mse)
rmse

#pour avoir une idée sur la performance du modele on verifier le mae pour training set et le mae pour testing set
y_pred_train = regressor.predict(X_train)

from sklearn.metrics import  mean_absolute_error

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_train
mae_test = mean_absolute_error(y_test, y_pred)
mae_test


#ici on va essayer de trouver un modele mais avec moins de variables, on va esayer 3 puis 4 puis 5
from sklearn.feature_selection import RFE

rfe_5 = RFE(regressor, n_features_to_select=5)
rfe_5.fit(X_train, y_train)
y_pred = rfe_5.predict(X_test)
print(r2_score(y_test, y_pred))

for i in range(X_train.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe_5.support_[i], rfe_5.ranking_[i]))

rfe_4 = RFE(regressor, n_features_to_select=4)
rfe_4.fit(X_train, y_train)
y_pred = rfe_4.predict(X_test)
print(r2_score(y_test, y_pred))

for i in range(X_train.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe_4.support_[i], rfe_4.ranking_[i]))
    



 


    




