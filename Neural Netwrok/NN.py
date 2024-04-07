!pip install tflearn

import numpy as np
import pandas as pd
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

from tflearn.datasets import titanic
dataset = ('titanic_dataset.csv')

# Read the content inside titanic dataset
dataframe = pd.read_csv(dataset)

# Select relevant columns
selected_columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'survived']
dataframe = dataframe[selected_columns]

# Convert 'Sex' to numerical values
dataframe['sex'] = dataframe['sex'].map({'male': 0, 'female': 1})

# Drop rows with missing values
dataframe = dataframe.dropna()

# Separate features and labels
X = dataframe.drop(columns=['survived'])
y = dataframe['survived'].values.reshape(-1, 1)

# Preprocessing function
def preprocess(X):
    # Normalize features
    X_normalized = (X - X.mean()) / X.std()
    # Add bias term
    X_bias = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]
    return X_bias

# Preprocess the features
X_processed = preprocess(X)

# Define neural network architecture
net = input_data(shape=[None, X_processed.shape[1]])
net = fully_connected(net, 5, activation='relu')  # Number of hidden units can be adjusted (e.g., 2, 3, 4, 5)
net = fully_connected(net, 1, activation='sigmoid')
net = regression(net)

# Define model
model = tflearn.DNN(net)

# Train the model
model.fit(X_processed, y, n_epoch=10, batch_size=16, show_metric=True)

# Example predictions
farhan = np.array([3, 0, 19, 0, 0, 5.0000]).reshape(1, -1)
hania = np.array([1, 1, 17, 1, 2, 100.0000]).reshape(1, -1)
shamil = np.array([2, 0, 25, 1, 1, 30.0000]).reshape(1, -1)
hamna = np.array([3, 1, 22, 0, 0, 7.2500]).reshape(1, -1)

# Preprocess data
farhan_processed = preprocess(farhan)
hania_processed = preprocess(hania)
shamil_processed = preprocess(shamil)
hamna_processed = preprocess(hamna)

# Predict surviving chances
pred_farhan = model.predict(farhan_processed)
pred_hania = model.predict(hania_processed)
pred_shamil = model.predict(shamil_processed)
pred_hamna = model.predict(hamna_processed)

print("Farhan's Surviving Rate:", pred_farhan[0][0])
print("Hania's Surviving Rate:", pred_hania[0][0])
print("Shamil's Surviving Rate:", pred_shamil[0][0])
print("Hamna's Surviving Rate:", pred_hamna[0][0])
