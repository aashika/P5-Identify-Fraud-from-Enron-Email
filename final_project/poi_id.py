#!/usr/bin/python

import sys
import pickle
import pprint
sys.path.append("../tools/")



from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.grid_search import GridSearchCV


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] # You will need to use more features


###The additional features would be added later

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

### After analyzing the insiderpaypdf, two outliers can be spotted: 'THE TRAVEL AGENCY IN THE PARK' and 'TOTAL'
### Further analysis to find additional outliers in case of dupicate or fake entries
names = []
for i in data_dict:
    names.append(i)

names.sort()
pprint.pprint(names)

## The names are printed so that they can be cross verified against the pdf, and duplicates are removed.

### 'THE TRAVEL AGENCY IN THE PARK' and 'TOTAL' are the only outliers and can be removed
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


### Adding two new features:
### fraction from poi =
### fraction to poi =
for j in my_dataset:
	employee = my_dataset[j]
	if (employee['from_poi_to_this_person'] != 'NaN' and
             employee['from_this_person_to_poi'] != 'NaN' and
             employee['to_messages'] != 'NaN' and
             employee['from_messages'] != 'NaN'
             ):
	    fraction_from_poi = float(employee["from_poi_to_this_person"]) / float(employee["from_messages"])
	    employee["fraction_from_poi"] = fraction_from_poi
	    fraction_to_poi = float(employee["from_this_person_to_poi"]) / float(employee["to_messages"])
	    employee["fraction_to_poi"] = fraction_to_poi
	else:
	    employee["fraction_from_poi"] = employee["fraction_to_poi"] = 0


my_features = features_list + ['salary',
                                  'deferral_payments',
                                  'total_payments',
                                  'loan_advances',
                                  'bonus',
                                  'restricted_stock_deferred',
                                  'deferred_income',
                                  'total_stock_value',
                                  'expenses',
                                  'exercised_stock_options',
                                  'other',
                                  'long_term_incentive',
                                  'restricted_stock',
                                  'director_fees',
                                  'fraction_from_poi',
                                  'fraction_to_poi'
                                  ]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "Chosen features:", my_features

# Scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# K-best features
k_best = SelectKBest(k=5)
k_best.fit(features, labels)

results_list = zip(k_best.get_support(), my_features[1:], k_best.scores_)
results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
print "KBest features:", results_list

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn import tree
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='random')
'''
#Uncomment the code to use the Gaussian classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
'''

'''
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='minkowski',
           metric_params=None, n_neighbors=10, p=3, weights='distance')
'''

'''
#Uncomment the code in order to try tuning the parameters for the DTC
from sklearn import grid_search
parameters = {'min_samples_split':[2,4,6,8,10],
              'splitter': ('best','random'),
              'max_depth':[None,2,4,6,8,10]
              }
clf_s = grid_search.GridSearchCV(tree.DecisionTreeClassifier(), parameters).fit(features, labels)
print 'Best estimator:'
print clf_s.best_estimator_
'''



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, my_features)



# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
