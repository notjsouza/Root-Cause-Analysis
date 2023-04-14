# Imports
import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tensorflow import keras

# Loading Datasheet into a Pandas Dataframe
data = pd.read_csv("root_cause_analysis.csv")

# Testing Data Output
# print(data.dtypes)

data.head()
encoder = preprocessing.LabelEncoder()
data['ROOT_CAUSE'] = encoder.fit_transform(data['ROOT_CAUSE'])

# Converting Pandas Dataframe to NumPy Vector
np_symptom = data.to_numpy().astype(float)

# Extracting Feature Variable X
x_data = np_symptom[:, 1:8]

# Extracting Target Variable Y, Convert to One-Hot-Encoding
y_data = np_symptom[:, 8]
y_data = tf.keras.utils.to_categorical(y_data, 3)

# Split Training and Testing Data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

# Testing Output
# print("Shape of feature variables:", x_train.shape)
# print("Shape of target variable:", y_train.shape)

# Training parameters
EPOCHS = 20
BATCH_SIZE = 64
VERBOSE = 1
OUTPUT_CLASSES = len(encoder.classes_)
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

# Creating a Sequential Model
model = tf.keras.models.Sequential()

# Creating Dense Layers
model.add(keras.layers.Dense(N_HIDDEN, input_shape=(7,), name='Dense-Layer-1', activation='relu'))
model.add(keras.layers.Dense(N_HIDDEN, name='Dense-Layer-2', activation='relu'))
model.add(keras.layers.Dense(N_HIDDEN, name='Dense-Layer-3', activation='relu'))
model.add(keras.layers.Dense(OUTPUT_CLASSES, name='Dense-Layer-Final', activation='softmax'))

# Compiling the Model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Building the Model
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# Testing the Model Against Test Data
print("\nEvaluation against Test Dataset:\n")
model.evaluate(x_test, y_test)

# Creates a Prediction for One Test Case
prediction = np.argmax(model.predict([[1, 0, 0, 0, 1, 1, 0]]), axis=1)
print(encoder.inverse_transform(prediction))

# Creates Predictions for Multiple Test Cases
print(encoder.inverse_transform(np.argmax(model.predict(
    [[1, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 1, 1]]), axis=1)))
