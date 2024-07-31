#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
import numpy as np
data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here

train_feature, test_feature, train_label, test_label = train_test_split(features,labels, test_size=0.3, random_state=42)
clf = tree.DecisionTreeClassifier()
clf.fit(train_feature, train_label)
prediction = clf.predict(test_feature)
print(f"Number of peoples in the test set: {len(test_feature)}")
print(f"Number of POIs predicted: {sum(prediction)}")
print(f"If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be? Answer: {prediction.tolist().count(0)/len(test_feature)}")

#True Positive label count
count = 0
for i in range(0, len(prediction)):
    if (prediction[i]==1 and test_label[i]==1):
        count += 1
print(f"True Positive label count is {count}")

#Precision
from sklearn.metrics import precision_score
precision = precision_score(test_label, prediction)
print(f"Precision score:{precision}")

#Recall
from sklearn.metrics import recall_score
recall = recall_score(test_label, prediction)
print(f"Recall score:{recall}")


#Calculating in a given sample
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]


true_pos_count = sum(1 for i in range(0,len(predictions)) if predictions[i]==1 and true_labels[i]==1)
print(f"True positive count in given sample: {true_pos_count}")

true_neg_count = sum(1 for i in range(0,len(predictions)) if predictions[i]==0 and true_labels[i]==0)
print(f"True negative count in given sample: {true_neg_count}")

false_pos_count = sum(1 for i in range(0,len(predictions)) if predictions[i]==1 and true_labels[i]==0)
print(f"False positive count in given sample: {false_pos_count}")

false_neg_count = sum(1 for i in range(0,len(predictions)) if predictions[i]==0 and true_labels[i]==1)
print(f"False negative count in given sample: {false_neg_count}")

acc = clf.score(test_feature, test_label)
print(f"Accuracy: {acc}")




