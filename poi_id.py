#!/usr/bin/python

import sys
import pickle
import pprint
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.cross_validation import StratifiedShuffleSplit
from itertools import combinations



# Modified test classifier function to return scores.
# It is used to iteratively select the best features
def test_classifier_modified(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives\
        + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        return [accuracy, precision, recall]
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true \
        positive predicitons."






### Task 1: Select what features you'll use.

# feature list to be iterated to find the best combination
# feature list does not contain 'poi' here. It is added later during iteration
features_list = ['salary','bonus', 'restricted_stock',
                 'long_term_incentive', 'exercised_stock_options',
                 'from_this_person_to_poi', 'from_poi_to_this_person',
                 'fraction_from_poi'] 

# Final selected features
features_selected = [ 'poi', 'bonus', 'long_term_incentive', 
                       'from_poi_to_this_person']


# features used to plot histograms to find outliers
features_to_plot = [ 'bonus', 
                     'exercised_stock_options', 'expenses',
                     'from_messages', 'from_poi_to_this_person',
                     'from_this_person_to_poi', 'long_term_incentive',
                     'restricted_stock', 'salary', 'other',
                     'shared_receipt_with_poi', 'to_messages',
                     'total_payments', 'total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Knowing the dataset
print 'Number of data points:', len(data_dict)
print 'Number of features:', len(data_dict['LAY KENNETH L'])
print '\nExmaple datapoint'
pp=pprint.PrettyPrinter()
pp.pprint(data_dict['LAY KENNETH L'])
#pp.pprint(data_dict['PAI LOU L'])

# Number of POIs in the dataset
number_of_poi = 0
for key, value in data_dict.iteritems():
    if value['poi'] == True:
        number_of_poi = number_of_poi + 1

print '\nPOI:', number_of_poi
print 'Non-POI', len(data_dict) - number_of_poi

### Task 2: Remove outliers

## From the lectures, we alreay know that there is a datapoint TOTAL
data_dict.pop('TOTAL', 0)
print "\nRemoved outlier TOTAL\n"



# function to count NaN's in each variable
def number_of_nans(dic):
    Nan = {}
    for key, value in dic.iteritems():
        for k, v in value.iteritems():
            Nan[k] = 0
        break
    
    for key, value in dic.iteritems():
        for k, v in value.iteritems():
            if value[k] == 'NaN':
                Nan[k] = Nan[k] + 1
               
    return Nan
            
Nan = number_of_nans(data_dict)
print "NaN's in each feature:"
pp.pprint(Nan)



# We can see that there are NaN's in every feature.
# Notably, a lot of NaN's in 'deferral_payments', 'deferred_income',
# 'loan_advances', 'restricted_stock_deferred', 'director_fees',


# Ignoring the variables with lot of NaN's, we will see the distribution
# of the rest of variables to detect any outliers.
data_plot = featureFormat(data_dict, features_to_plot, sort_keys = True)
# histogram is a great tool for detecting outliers.
# salary = [item[1] for item in data]


f, axes = plt.subplots(7,2, figsize=(15,10))
l=0
for i in range(7):
    for j in range(2):
        axes[i,j].hist(data_plot[:,l])
        axes[i,j].set_title(features_to_plot[l])
        l = l + 1
plt.tight_layout()



# Getting top 6 values for each varible along with label POI.
def top_six_values(dic, feature):
    # removing NaNs    
    to_remove = []    
    for k, v in dic.iteritems():
        if v[feature]=='NaN':
            to_remove.append(k)
    for i in to_remove:
        dic.pop(i)
    # sorting  
    feature_list = [v[feature] for k, v in dic.iteritems() ]
    poi_list = [v['poi'] for k, v in dic.iteritems() ]
    name_list = [k for k, v in dic.iteritems()]
    sorted_feature, sorted_poi, sorted_name = zip(*sorted(zip(feature_list,
                                                              poi_list,
                                                              name_list),
                                                          reverse = True))
    print sorted_name[:6]
    print sorted_poi[:6]

# top_six_values(data_dict, 'salary')
# top_six_values(data_dict, 'bonus')      #LAVORATO JOHN J
# top_six_values(data_dict, 'expenses')
# top_six_values(data_dict, 'exercised_stock_options')
# top_six_values(data_dict, 'from_messages')                          
# top_six_values(data_dict, 'from_poi_to_this_person')
# top_six_values(data_dict, 'from_this_person_to_poi')
# top_six_values(data_dict, 'long_term_incentive')
# top_six_values(data_dict, 'restricted_stock')
# top_six_values(data_dict, 'shared_receipt_with_poi')
# top_six_values(data_dict, 'to_messages')
# top_six_values(data_dict, 'total_payments')
# top_six_values(data_dict, 'total_stock_value')

# We can see most of the outliers are real POIs.
# There are no outliers like total.



# Manual inspection of keys
'''
for key, value in data_dict.iteritems():
    print key
'''
# Found a wrong entry THE TRAVEL AGENCY IN THE PARK. Defenitely not a person.
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
print "\nRemoved wrong entry 'THE TRAVEL AGENCY IN THE PARK'"



# From the histogram, we can see that there are negative values present for 
# resticted_stock and total_stock_value which is not possible.
# We will investigate it and compare it with PDF data  to verify
'''
for key , value in data_dict.iteritems():
    if value['restricted_stock'] < 0 or value['total_stock_value'] < 0:
        print key
        pp.pprint(data_dict[key])
'''
# After comparing it with the financial and salary data from PDF document,
# We find that they are wrong entries. It is now corrected.
data_dict['BELFER ROBERT']['deferral_payments']='NaN'
data_dict['BELFER ROBERT']['deferred_income']=-102500
data_dict['BELFER ROBERT']['director_fees']=102500
data_dict['BELFER ROBERT']['exercised_stock_options']='NaN'
data_dict['BELFER ROBERT']['expenses']=3285
data_dict['BELFER ROBERT']['restricted_stock']=44093
data_dict['BELFER ROBERT']['restricted_stock_deferred']=-44093
data_dict['BELFER ROBERT']['total_payments']=3285
data_dict['BELFER ROBERT']['total_stock_value']='NaN'

data_dict['BHATNAGAR SANJAY']['director_fees']='NaN'
data_dict['BHATNAGAR SANJAY']['exercised_stock_options']=15456290
data_dict['BHATNAGAR SANJAY']['expenses']=137864
data_dict['BHATNAGAR SANJAY']['other']='NaN'
data_dict['BHATNAGAR SANJAY']['restricted_stock']=2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred']=-2604490
data_dict['BHATNAGAR SANJAY']['total_payments']=137864
data_dict['BHATNAGAR SANJAY']['total_stock_value']=15456290

print "\nCorrected errors in entries for 'BELFER ROBERT' and 'BHATNAGAR \
SANJAY'"



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
for key, value in data_dict.iteritems():
    if data_dict[key]['from_poi_to_this_person'] != 'NaN' or \
       data_dict[key]['from_messages'] != 'NaN':
           data_dict[key]['fraction_from_poi'] = \
           data_dict[key]['from_poi_to_this_person'] \
           /data_dict[key]['from_messages']
    else:
        data_dict[key]['fraction_from_poi'] = 0
    
print "Added new feature 'fraction_from_poi'"


my_dataset = data_dict







### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# checking for best possible combinations of features from 
# the features_list selected through case study knowledge.
# Two algorithms are used. Gaussian Naive Bayes and Decision Trees
'''
score_list= {}
for i in range(0,8):
    for j in combinations(features_list, i+1):
        lis = []        
        f = ['poi'] + list(j) # 'poi' is always first feature
        #clf = GaussianNB()
        clf = DecisionTreeClassifier()
        # modified test_classifier function returns evaluation metrics
        # to be stored in a dictionary instead of just printing them.
        score_list["+".join(f)] = test_classifier_modified(clf, my_dataset, f)

print "Done"
# format of returned metrics = [accuracy, precision, recall]        
# filtering for combinations of features with accuracy > 0.8
# precision > 0.3 and recall > 0.3

score_list_sorted_keys = sorted(score_list, key = lambda x: score_list[x][2], reverse = True)
j=0
for i in score_list_sorted_keys:
    print i, score_list[i]
    if j == 4:
        break
    j = j + 1
# from this we find that the best combination for high recall
# precision and accuray is found using just 3 variables
# 'salary','exercised_stock_options' and 'fraction_from_poi'.
        

# SVM was painfully slow to fit and predict, so avoided it from the analysis
'''








### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

data = featureFormat(my_dataset, features_selected , sort_keys = True)
labels, features = targetFeatureSplit(data)
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)

# Setting the parameters to be varied
parameters = { 'criterion': ('gini', 'entropy'), 'splitter':('best', 'random'),
                'min_samples_split':(2,5,10)}
                

# Searching for the best parameters and printing them               
clf_simple = DecisionTreeClassifier()
# maximizing the recall score
search_param = GridSearchCV(clf_simple, parameters, scoring = 'recall',
                            cv = cv)
search_param.fit(features, labels)
print "\n\n Best parameters are:", search_param.best_params_

# Using the best parameters
clf = clf_simple.set_params(**search_param.best_params_)



test_classifier(clf, my_dataset, features_selected)

#print 'Feature importances of decision tree classifier:', \
 #clf.feature_importances_





### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_selected)
