import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
catData = genfromtxt('categoryFeatures.csv', delimiter=',', dtype=str, encoding='UTF-8')
labels = genfromtxt('normedLabels.csv', delimiter=',')

contData = genfromtxt('contFeaturesNormed.csv', delimiter=',')

#remove first entry, is bugged for some reason
contData = np.delete(contData, 0,0)
catData = np.delete(catData, 0,0)
labels = np.delete(labels, 0,0)

#need to encode the categorical data
#gets the unique values for each column as lists
entry0 = np.unique(catData[:,0])
entry1 = np.unique(catData[:,1])
entry2 = np.unique(catData[:,2])
entry3 = np.unique(catData[:,3])
entry4 = np.unique(catData[:,4])
entry5 = np.unique(catData[:,5])
entry6 = np.unique(catData[:,6])
entry7 = np.unique(catData[:,7])
#set up the encoder, replace each cat column with its respective encoding
le = preprocessing.LabelEncoder()
#fits the encoder to use string values of the list we want it to
le.fit(entry0)
#transforms the strings to ints using the fitted values given before
catData[:,0] = le.transform(catData[:,0])

le = preprocessing.LabelEncoder()
le.fit(entry1)
catData[:,1] = le.transform(catData[:,1])

le = preprocessing.LabelEncoder()
le.fit(entry2)
catData[:,2] = le.transform(catData[:,2])

le = preprocessing.LabelEncoder()
le.fit(entry3)
catData[:,3] = le.transform(catData[:,3])

le = preprocessing.LabelEncoder()
le.fit(entry4)
catData[:,4] = le.transform(catData[:,4])

le = preprocessing.LabelEncoder()
le.fit(entry5)
catData[:,5] = le.transform(catData[:,5])

le = preprocessing.LabelEncoder()
le.fit(entry6)
catData[:,6] = le.transform(catData[:,6])
le = preprocessing.LabelEncoder()
le.fit(entry7)
catData[:,7] = le.transform(catData[:,7])
#normalize these new values, and convert them to floats because they are number strings
catData = catData.astype('float64')
catData = catData / catData.max(axis=0)


data = np.append(contData[:,0].reshape(contData[:,0].shape[0],1),catData[:,[0,2]],1)
data = np.append(data, contData[:,1].reshape(contData[:,0].shape[0],1),1)
data = np.append(data, catData[:,[5,6,7]],1)
data = np.append(data, contData[:,2].reshape(contData[:,0].shape[0],1),1)


from sklearn.model_selection import train_test_split
X = data
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(13,13,13),max_iter=500)

print(y_train.dtype)
y_train = y_train.astype('float64')
print(X_train.dtype)
mlp.fit(X_train,y_train)
MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
y_pred = mlp.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('R2 score:', mlp.score(X_test, y_test))
plt.scatter(y_test, y_pred, color='blue')
x=np.linspace(-0.1,1,1000)
plt.plot(x, x, color='red')
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()
