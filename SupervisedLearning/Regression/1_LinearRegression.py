import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Listo, entonces una vez hayamos analizado los datos vamos a alimentar nuestro modelo usando esta informacion
boston = pd.read_csv('Boston.csv')
X = boston.drop('medv', axis=1).values
y = boston['medv'].values
X_rooms = X[:, 6]
#La API de scikit learn exige que presentemos los datos en forma de matrices así que procedemos a añadirle otra dimensión
# a nuestros set de datos usando el método reshape
y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)

#usamos un modelo llamado LinearRegression
reg = LinearRegression()
#y tal cual como hicimos con el ejemplo de Clasificacion, le pasamos los datos usando el metodo fit
#le pasamos el numero de habitaciones, y la variable objetivo, el precio de la vivienda
reg.fit(X_rooms, y)
#linspace Return evenly spaced numbers over a specified interval.
pred_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1, 1)
plt.scatter(X_rooms, y, color='blue')
plt.plot(pred_space, reg.predict(pred_space), color='black', linewidth=3)
plt.show()

print('R^2: {}'.format(reg.score(X_rooms, y)))
