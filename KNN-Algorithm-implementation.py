#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import csv
import random


with open(r'C:\Users\LENOVO\Desktop\machineLearngin\iris.data') as csvfile:
    lines = csv.reader(csvfile)
        


# In[15]:



# spliting dataSet
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range (len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


# In[16]:


# test spliting

trainingSet=[]
testSet=[]
loadDataset(r'C:\Users\LENOVO\Desktop\machineLearngin\iris.data', 0.66, trainingSet, testSet)
print('train' + repr(len(trainingSet)))
print('test' + repr(len(testSet)))


# In[17]:


# find the similaritis (distance between any two data instances)
import math

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


# In[19]:


# test euclideanDistance function
# sqrt((4-2)^2 + (4-2)^2 + (5-2)^2 )
data1 = [2, 2, 2, 'a']
data2 = [4, 4, 5, 'b']
distance = euclideanDistance(data1, data2, 3)
print('distance is ' + repr(distance))


# In[26]:


# look for k nearest neigbors
# k belo is the nearest neigbors number you want to check for
import operator

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# In[30]:


# test above function

trainingSet=[[2, 2, 2, 'a'], [1, 1, 1, 'b']] # training data two classes a and b
testInstance=[5, 5, 5] # we need to know this test instance belong to class a or class b above
k = 1
neighbors = getNeighbors(trainingSet, testInstance, 1)
print(neighbors)


# In[37]:


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


# In[38]:


neighbors = [[2, 2, 2, 'a'], [1, 1, 1, 'a'], [3, 3, 3, 'b']]
response = getResponse(neighbors)
print(response)


# In[39]:


# evaluate the accuracy of KNN

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] in predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


# In[42]:


testSet = [[2, 2, 2, 'a'], [1, 1, 1, 'a'], [3, 3, 3, 'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print(accuracy)


# In[44]:


# now define a function for wholething

def main():
    # prepar data
    trainingSet=[]
    testSet=[]
    split = 0.67
    loadDataset(r'C:\Users\LENOVO\Desktop\machineLearngin\iris.data', split, trainingSet, testSet)
    print('train set ' + repr(len(trainingSet)))
    print('test set ' + repr(len(testSet)))
    # generate predictions
    predictions=[]
    k=3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('>predictions= ' + repr(result) + 'actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('accuracy=' +repr(accuracy) + '%')

main()


# In[ ]:




