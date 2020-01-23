import operator
import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self,k,weighted=False):
        self.k=k
        self.weighted=weighted

    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        print("Training done")

    def predict_normal(self,X_test):
        distance={}
        counter=0
        n=self.X_train.shape[1]
        classification=np.array([],dtype='int')
        for i in X_test:
            p=0
            for j in self.X_train:
                p=p+np.sum((i-j) ** 2)
                p=p**1/2
                distance[counter]=p
                counter=counter+1
            distance=sorted(distance.items(),key=operator.itemgetter(1))
            m=self.classify(distance[0:self.k])
            classification=np.append(classification,m)
            counter=0
            distance={}
        return classification

    def predict_weighted(self,X_test):
        distance={}
        counter=0
        n=self.X_train.shape[1]
        classification=np.array([],dtype='int')
        for i in X_test:
            p=0
            for j in self.X_train:
                p=p+np.sum((i-j) ** 2)
                p=p**1/2
                distance[counter]=p
                counter=counter+1
            distance=sorted(distance.items(),key=operator.itemgetter(1))
            m=self.classify(distance[0:self.k])
            w=distance[0:self.k]
            print(w)
            classification=np.append(classification,m)
            counter=0
            distance={}
        return classification

    def classify(self,distance):
        label=[]
        for i in distance:
            label.append(self.y_train[i[0]])
            print(self.y_train[i[0]])
        return Counter(label).most_common()[0][0]

    def weightedPredict(self, testing_data):
        distance = {}
        counter = 0
        classification = np.array([], dtype='int')

        for i in testing_data:
            for j in self.X_train:
                distance[counter] = np.sqrt(np.sum((i - j) ** 2))
                counter = counter + 1
            distance = sorted(distance.items(), key=lambda x: (x[1], x[0]))
            classification = np.append(classification, [[self.weightedClassify(i, distance[0:self.k])]])
            distance = {}
            counter = 0
        return classification

    def weightedClassify(self, test_data, distance):
        label = {}
        for i in distance:
            label[self.y_train[i[0]]] = label.get(self.y_train[i[0]], 0) + (1 / i[1])
        label = sorted(label.items(), key=lambda x: (x[1], x[0]), reverse=True)
        return label[0][0]

    def predict(self,test_data):
        if(self.weighted):
            return self.predict_normal(test_data)
        else:
            return self.predict_weighted(test_data)