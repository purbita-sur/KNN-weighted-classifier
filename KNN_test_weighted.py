import numpy as np
import pandas as pd
from KNearestNeighbor import KNearestNeighbors

data=pd.read_csv("Social_Network_Ads.csv")
X=data.iloc[:,2:4].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

print(X_train)

def normal_input():
    k = int(input('Enter K value: '))
    knn = KNearestNeighbors(k=k)
    knn.fit(X_train=X_train, y_train=y_train)
    answer = knn.predict(X_test).tolist()
    for i in range(0, len(answer)):
        if answer[i] == 0:
            answer[i] = 'Will not Purchase'
        elif answer[i] == 1:
            answer[i] = 'Will purchase'
        print(answer[i])

def weighted_input():
    k = int(input('Enter K value: '))
    knn = KNearestNeighbors(k=k,weighted=True)
    knn.fit(X_train=X_train,y_train=y_train)
    answer = knn.predict(X_test).tolist()
    for i in range(0, len(answer)):
        if answer[i] == 0:
            answer[i] = 'Will not Purchase'
        elif answer[i] == 1:
            answer[i] = 'Will purchase'
        print(answer[i])

choice = int(
    input(
        '1. Choose 1 for normal KNN Input\n2. Choose 2 for weighted KNN\n'))
if choice == 1:
    normal_input()
if choice == 2:
    weighted_input()