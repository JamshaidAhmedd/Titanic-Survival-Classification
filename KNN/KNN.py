!pip install tflearn

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
data = pd.read_csv("titanic.csv")

# Drop uninformative features including 'name'
columns_to_drop = ['PassengerId', 'name', 'ticket', 'Cabin', 'Embarked']
data.drop(columns=columns_to_drop, inplace=True)


# Normalize numerical features
numerical_columns = ['age', 'sibsp', 'parch', 'fare']
data[numerical_columns] = (data[numerical_columns] - data[numerical_columns].min()) / (data[numerical_columns].max() - data[numerical_columns].min())

# Encode categorical variable 'sex'
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])

# Drop rows with NaN values
data.dropna(inplace=True)

# Extract features (X) and target variable (y)
X = data.drop('survived', axis=1)
y = data['survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the K-Nearest Neighbors Model
k = 5  #We can use any number for k here
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

# Evaluate the performance of the KNN model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize decision boundary using Principal Component Analysis (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_pca, y)

# Create a meshgrid to plot the decision boundary
h = 0.02
x_min, x_max = X_pca[:, 0].min() - 0.1, X_pca[:, 0].max() + 0.1
y_min, y_max = X_pca[:, 1].min() - 0.1, X_pca[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on the meshgrid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'KNN Decision Boundary (k={k}) - PCA Reduced')
plt.show()

# Experiment with different numbers of neighbors and plot accuracy
neighbors = np.arange(1, 100)
accuracy_scores = []

for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Plot the accuracy scores
plt.plot(neighbors, accuracy_scores)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Neighbors')
plt.show()
