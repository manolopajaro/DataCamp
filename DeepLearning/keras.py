# todo install keras

from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import keras
from keras.layers import Dense
from keras.models import Sequential

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

my_optimizer = SGD(lr=0.01)  # learning rate
# Compile the model
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer=my_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping_monitor = EarlyStopping(patience=2)
# Fit the model
model.fit(X, y, validation_split=0.3, callbacks=[early_stopping_monitor])
