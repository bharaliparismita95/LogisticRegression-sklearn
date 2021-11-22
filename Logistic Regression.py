import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading data
data = pd.read_csv('iris_data.csv')
print(data.head())

# Splitting data into x and y(label)
x = data.iloc[:, 0: 4].values
y = data.iloc[:, 4].values

# Splitting the data into Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Training model on train data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)

# Testing model on test data
prediction = model.predict(x_test)

# Getting the weights and bias
w = np.array(model.coef_)
b = model.intercept_
print('---------Weights and Bias----------')
print('weights:', w)
print('bias:', b)

# Getting the training score and test score for the model
print('----------Train & Test Accuracy-----------')
print('Train accuracy:', model.score(x_train, y_train))
print('Test accuracy:', model.score(x_test, y_test))

# Printing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)
print('-----------Confusion Matrix------------')
print(cm)

# Plotting the confusion matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model, x_test, y_test)
plt.show()
