!pip install tflearn
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tflearn.datasets import titanic
dataset = ('titanic_dataset.csv')

# Read the dataset
dataframe = pd.read_csv(dataset)

# Load CSV file, first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv(dataset, target_column=0, categorical_labels=True, n_classes=2)

# Data preprocessing - Convert the 'sex' feature into numerical value
def preprocess(data, discarded_columns):
    for id in sorted(discarded_columns, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
        data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

# Columns to discard in preprocessing
discarded_columns = [1, 6]

# Preprocess data
data = preprocess(data, discarded_columns)

# Define a function to create and train a neural network model
def train_neural_network(initializer, data, labels, learning_rate, num_epochs=20, batch_size=16):
    # Reset the TensorFlow graph
    tf.compat.v1.reset_default_graph()

    # Build neural network classifier
    net = tflearn.input_data(shape=[None, 6])
    net = tflearn.fully_connected(net, 32, weights_init=initializer())
    net = tflearn.fully_connected(net, 32, weights_init=initializer())
    net = tflearn.fully_connected(net, 2, activation='softmax', weights_init=initializer())
    net = tflearn.regression(net, learning_rate=learning_rate)

    # Define model
    model = tflearn.DNN(net)

    # Start training
    history = model.fit(data, labels, n_epoch=num_epochs, batch_size=batch_size, show_metric=True, snapshot_epoch=False)

    # Return the trained model and training history
    return model, history

# Train model with zero weights initialization
zero_initializer = tf.zeros_initializer
model_zero, history_zero = train_neural_network(zero_initializer, data, labels, learning_rate=0.01)

# Train model with small random weights initialization
small_random_initializer = tf.random_normal_initializer(stddev=0.01)
model_random, history_random = train_neural_network(small_random_initializer, data, labels, learning_rate=0.01)

# Plot iterations vs. loss graphs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_zero.epoch, history_zero.history['loss'], label='Zero Initialization')
plt.title('Zero Initialization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_random.epoch, history_random.history['loss'], label='Random Initialization')
plt.title('Random Initialization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
