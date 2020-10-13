import pandas as pd
import matplotlib.pyplot as plt

boston = pd.read_csv('Boston.csv')
print(boston.head())
#CRIM tasa de criminalidad per capita, NX concentracion de oxidos nitricos , RM numero promedio de habitaciones
#nuestra variable objetivo, MEDV, es el valor medio en miles de dolares de las viviendas ocupadas por propietarios
# separamos los datos en nuestras variables objetivo y caracteristicas
X = boston.drop('medv', axis=1).values
y = boston['medv'].values

# Como primer ejemplo vamos a intentar predecir el precio de una casa basado en una sola caracteristica, el número promedio de habitaciones
X_rooms = X[:, 6] #tomamos el valor de la quinta columna, rm, correspondiente al número de habitaciones y creamos un set de variables predictoras listas para nuestro modelo

#esta parte de reshape es necesaria para convertir nuestros arrays a matrices, forma que es necesaria para el modelo
y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)

#y para darnos un poco de dimension sobre la relacion entre estos dos datos, vamos a hacer un grafico usando una libreria de python llamada pyplot
plt.scatter(X_rooms, y)
plt.ylabel('Valor casa en miles de dolares')
plt.xlabel('numero de habitaciones')
plt.show()
#de forma inmediata podemos ver que obviamente, entre mas habitaciones tenga una casa, mas cara va a ser
