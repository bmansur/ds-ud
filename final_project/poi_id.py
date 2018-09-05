#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn import cross_validation 
import numpy as np


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','deferral_payments','total_payments','loan_advances','bonus',
'restricted_stock_deferred','deferred_income','total_stock_value','expenses','exercised_stock_options',
'other','long_term_incentive','restricted_stock','director_fees','to_messages','from_poi_to_this_person',
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

print "Numero de features a serem utilizados: ", len(features_list) - 1

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "Numero de dados no dataset: ", len(data_dict)

df = pd.read_pickle("final_project_dataset.pkl")

### Task 2: Remove outliers
del data_dict['TOTAL']

### Task 3: Create new feature(s)
for data in data_dict.values():
    data['to_poi_msg_ratio'] = 0
    if float(data['from_messages']) > 0:
        data['to_poi_msg_ratio'] = float(data['from_this_person_to_poi'])/float(data['from_messages'])

#Remover o comentario para adicionar as novas features
features_list.extend(['to_poi_msg_ratio'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
## For local testing
def resultado(algorithm, clf):
    # Provided to give you a starting point. Try a variety of classifiers.
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, random_state=42)
    clf.fit(features_train, labels_train)
    print 'Algoritmo: ', algorithm
    print 'Melhor score: %0.3f' % clf.best_score_
    print 'Melhor parameters set:'
    best_parameters = clf.best_estimator_.get_params()
    new_params = {}
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
        new_params[param_name] = best_parameters[param_name]
    predictions = clf.predict(features_test)
    print 'Accuracy: ', accuracy_score(labels_test,predictions)
    print 'Recall: ', recall_score(labels_test,predictions)
    print 'Precision: ', precision_score(labels_test,predictions)

# Provided to give you a starting point. Try a variety of classifiers.
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
'''
kbest = SelectKBest(k = 6)
kbest.fit(features, labels)
f_list = zip(kbest.get_support(), features_list[1:], kbest.scores_)
f_list = sorted(f_list, key=lambda x: x[2], reverse = True)

import pprint
print "K-best features:", 
pprint.pprint(f_list)
'''
#updated feature list
features_list = ['poi','exercised_stock_options','total_stock_value','bonus','salary','to_poi_msg_ratio','deferred_income']
#features_list = ['poi','exercised_stock_options','total_stock_value','bonus','salary','deferred_income','long_term_incentive']

'''
kNeighbors = KNeighborsClassifier()
parameters = {'n_neighbors': [2,3,4,5,6,7],
                      'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                      'weights': ['uniform', 'distance'],
                      'p': [2,3,4,5,6,7,8]}
clf = GridSearchCV(kNeighbors, parameters, verbose=1, cv=10)
resultado("kNeighbors", clf)

tree = DecisionTreeClassifier()
parameters = {'criterion': ["gini", "entropy"],
              'splitter': ['best', 'random'],
              'min_samples_split': [3,4,5]
             }
clf = GridSearchCV(tree, parameters, verbose=1, cv=10)
resultado("DecisionTreeClassifier", clf)

bayes =  GaussianNB()
parameters = {}
clf = GridSearchCV(bayes, parameters, verbose=1, cv=10)
resultado("GaussianNB", clf)

adaBoost = AdaBoostClassifier()
parameters = {'algorithm': ["SAMME", "SAMME.R"],
              'random_state': [1000,1200,1400,1600]
            }
clf = GridSearchCV(adaBoost, parameters, verbose=1, cv=10)
resultado("AdaBoostClassifier", clf)
'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#clf = KNeighborsClassifier(algorithm="ball_tree", n_neighbors=3, p=3, weights="uniform")
#clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=3, splitter='random')
#clf = AdaBoostClassifier(algorithm='SAMME', random_state=1000)
clf = GaussianNB(priors=None)
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)