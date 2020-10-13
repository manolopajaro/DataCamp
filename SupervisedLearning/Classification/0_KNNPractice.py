
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#Todos los modelos de ML en python estan implementados como clases que implementan los algoritmos para aprender y predecir
# y ademas guardan la informacion que aprenden después de entrenarse

#Por ejemplo, en python todos los modelos usan el metodo fit() para entrenar un modelo y usan el metodo predict para predecir
# la etiqueta de los datos sin etiquetar. Ven que es facil?

#Vamos entonces a iniciar importando nuestro clasificador para poder usarlo con
from sklearn.neighbors import KNeighborsClassifier
#y luego lo instanciamos indicando el numero de vecinos que queremos usar en esta ocasion
knn = KNeighborsClassifier(n_neighbors=6)
#knn = KNeighborsClassifier(n_neighbors=4)

#la data que vamos a usar para nuestro ejemplo la vamos a sacar de uno de los modulos de sklearn y se importa con
from sklearn import datasets
#luego se lo asignamos a una variable para que lo podamos usar
iris = datasets.load_iris()
#type(iris) es Blunt, lo cual es similar a un Dictionary para los que trabajan en desarrollo, y para los que no,
# imaginen un conjunto de datos agrupados en parejas de laforma clave-valor

#0 Setosa, 1 Versicolour, and 2 Virginica
#TODO desglosa un poco qué contiene la variable iris

#y con esto ya podemos empezar a entrenar nuestro modelo usando las caracteristicas, o variables predictoras,
# y nuestra variable objetivo
data = iris['data']
target = iris['target']
#analizando los datos que vamos a usar para entrenar nuestro modelo podemos apreciar que ambos arreglos son matrices,
# donde cada columna es una caracteristicas y cada fila es una muestra. Esto lo hace apto para nuestro modelo ya que
# para que se pueda entrenar nuestro modelo, debe tener esta, entre otras, caracteristicas.
#Vemos entonces que nuestra data tiene 150 muestras con cuatro caracteristicas cada una y
# target debe tener 150 observaciones, igual que data, pero con una sola columna
knn.fit(data, target)

#cable aclarar que este ejemplo muestra el funcionamiento básico usando datos que ya vienen cargados en la herramienta,
# en realidad la API de scikit nos exije que le presentemos los datos con ciertas caracteristicas para que los pueda procesar.

#Estas caracteristicas son:
#Que los arreglos de datos esten organizando en matrices
#Que las caracteristicas, o variables predictoras, contengan valores continuos (como el precio de una casa )y no sean etiquetas (como Masc y Fem)
#Que no falten valores en los datos
#Todos los sets de datos con los que vamos a trabajar hoy cumplen con estas caracteristicas y el tema del
# preprocesamiento de datos necesario para poder hacerlos aptos para un modelo cualquiera, y tratar temas como datos faltantes es toda otra charla


#Habiendo entrenado nuestro modelo, ya podemos empezar a usarlo para hacer predicciones y determinar las etiquetas de
# datos sin etiquetar. Sin embargo, ahora mismo no tenemos datos sin etiqueta ya que todos los que teniamos los usamos para entrenar el modelo!

# Vamos entonces a crear un set de muestras que podamos usar para preguntar a nuestro modelo que etiqueta le pondria
#en este caso voy a crear un set de 3 observaciones, cada una con sus respectivas cuatro caracteristicas
X_new = np.array([[5.6, 3.8, 5.1, 2.1],
                  [5.7, 2.6, 3.8, 1.3],
                  [4.7, 3.2, 1.3, 0.2]])

#usamos el metodo predict
prediction = knn.predict(X_new)
#y como podemos ver, el modelo nos arroja una prediccion por cada muestra que le pasamos
print('Prediccion: {}'.format(prediction))



