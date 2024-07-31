#!/usr/bin/python

import sys
import pickle
import os
sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np


### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
with open("../final_project/final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


financial_features = [
    'salary', 'deferral_payments', 'total_payments', 'loan_advances',
    'bonus', 'restricted_stock_deferred', 'deferred_income',
    'total_stock_value', 'expenses', 'exercised_stock_options',
    'other', 'long_term_incentive', 'restricted_stock', 'director_fees'
]

email_features = [
    'to_messages', 'from_poi_to_this_person', 'from_messages',
    'from_this_person_to_poi', 'shared_receipt_with_poi'
]

# Combine financial and email features
combined_features = financial_features + email_features

features_list = ['poi'] + combined_features



### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)  #This function converts the dataset dictionary into a numpy array of features.
labels, features = targetFeatureSplit(data)  #This function separates the target (label) from the features.

### Task 1: Select what features you'll use.
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)
selector.fit(features, labels)
features = selector.transform(features)

# Get the selected feature names
selected_features = [features_list[i] for i in selector.get_support(indices=True)]
selected_features = ['poi', 'salary'] + selected_features
print(f"Selected features: {selected_features}")

data = featureFormat(my_dataset, selected_features, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 2: Remove outliers
def remove_outlier(prediction, train_feature, train_label):
    cleaned_data = []
    error = abs(prediction - train_label)
    error_list = list(
        zip(train_feature, train_label, error))  # zip function combines multiple feature into a single tuple iteration

    error_list.sort(key=lambda x: x[2])

    reminder_num = int(len(error_list) * 0.9)

    cleaned_data = error_list[:reminder_num]

    return cleaned_data


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf = SVC()
clf.fit(features_train, labels_train)
prediction_svc = clf.predict(features_train)

print(f"Accuracy before removing outliers: {clf.score(features_train,labels_train)}")

cleaned_data = remove_outlier(prediction_svc, features_train, labels_train)

# Extract cleaned features and labels
cleaned_features = [data[0] for data in cleaned_data]  #cleaned_data =[features, labels, error]
cleaned_labels = [data[1] for data in cleaned_data]

# Refit the classifier with cleaned data
clf.fit(cleaned_features, cleaned_labels)
print(f"Accuracy after removing outliers: {clf.score(features_test, labels_test)}")




### Task 3: Create new feature(s)
# Create new features and add them to the dictionary
for key, data in data_dict.items():
    to_messages = float(data.get('to_messages', 1))  #If the key is null, default value is 1 instead of 0 to avoid dividing by 0
    from_messages =float(data.get('from_messages', 1))

    # Calculate ratios
    data['to_poi_ratio'] = float( data.get('from_poi_to_this_person', 0) )/ to_messages
    data['from_poi_ratio'] = float(data.get('from_this_person_to_poi', 0)) / from_messages
    data['shared_poi_ratio'] = float(data.get('shared_receipt_with_poi', 0)) / to_messages




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


##Feature Scaling
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)


##Finding best classifier
classifiers = {
'GaussianNB': GaussianNB(),
    'SVC': SVC(class_weight='balanced', random_state=42),
    'RandomForestClassifier': RandomForestClassifier(class_weight='balanced', random_state=42),
    'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(class_weight='balanced', random_state=42)
}

results = {}

for classifier_name, clf in classifiers.items():
    clf.fit(features_train, labels_train)
    clf.predict(features_test)

    accuracy = clf.score(features_test,labels_test)

    # Store the results
    results[classifier_name] = accuracy

# Print the results
for clf_name, accuracy in results.items():
    print(f"{clf_name}: {accuracy:.4f}")


best_clf_name = max(results, key=results.get) #Selects the best classifier based on accuracy (key = results.get)
best_clf = classifiers[best_clf_name]

print(f"Best classifier: {best_clf_name}")




### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



from sklearn.metrics import classification_report
prediction = best_clf.predict(features_test)
print(classification_report(labels_test,prediction))






### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = clf
my_dataset = data_dict
features_list_ = ['poi'] + combined_features

# Save the classifier, dataset, and features list
dump_classifier_and_data(clf, my_dataset, features_list_)

