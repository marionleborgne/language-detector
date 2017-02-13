"""Build a language detector model

The goal of this exercise is to train a linear classifier on text features
that represent sequences of up to N consecutive characters so as to be
recognize natural languages by using the frequencies of short character
sequences as 'fingerprints'.

The script saves the trained model to disk for later use
"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD
# Adapted by: Francesco Mosconi
import numpy as np
import itertools
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score)
from sklearn.preprocessing import StandardScaler


# The training data folder must be passed as first argument
try:
    dataset = load_files('./wikidata/short_paragraphs')
except OSError as ex:
    print ex
    print "Couldn't import the data, try running `python fetch_data.py` first "
    exit(-1)

# TASK: Split the dataset in training and test set
# (use 20% of the data for test):

features = dataset['data']
label = dataset['target']
feature_names = dataset['target_names']

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    label,
                                                    test_size=.2,
                                                    random_state=0)

# TASK: Build a an vectorizer that splits strings into sequence of 1 to 3
# characters instead of word tokens using the class TfidfVectorizer
transformer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), 
                              use_idf=True)

# TASK: Use the function make_pipeline to build a
#       vectorizer / classifier pipeline using the previous analyzer
#       and a classifier of choice.
#       The pipeline instance should be stored in a variable named model
clf = LogisticRegression(C=100)
scaler = StandardScaler()
model = make_pipeline(transformer, clf)

# TASK: Fit the pipeline on the training set
model.fit(X_train, y_train)

# TASK: Predict the outcome on the testing set.
# Store the result in a variable named y_predicted
y_pred = model.predict(X_test)
print 'Test accuracy: %.4f' % accuracy_score(y_test, y_pred) 

# TASK: Print the classification report
print classification_report(y_test, y_pred)

# TASK: Print the confusion matrix
print '=======confusion matrix=========='
cm = confusion_matrix(y_test, y_pred)

# TASK: Is the score good? Can you improve it changing
#       the parameters or the classifier?
#       Try using cross validation and grid search

# TASK: Use dill and gz to persist the trained model in memory.
#       1) gzip.open a file called my_model.dill.gz
#       2) dump to the file both your trained classifier
#          and the target_names of the dataset (for later use)
#    They should be passed as a list [clf, dataset.target_names]

import dill 
import gzip
with gzip.open('my_model.dill.gz', 'wb') as f:
    dill.dump([model, dataset.target_names], f)
    
    
