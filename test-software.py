import gensim
import pandas as pd
import numpy as np
import sys
import os
import itertools
import sklearn

from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from scipy import stats
from bs4 import BeautifulSoup


import helpers as helper
import pickle

import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")


"""This will use trained word2vec wikipedia models on dutch, english, spanish.
It will use word2vec with dimension 100."""
current_working_dir = os.getcwd() + '/'
model_dir = "word2vec-models/wikipedia-only-trained-on-my-machine/"

relations = {'dutch': {'truth_file': 'summary-dutch-truth.txt',\
                       'model_file': 'wiki.nl.tex.d100.model'
                       },
             'english': {'truth_file': 'summary-english-truth.txt',\
                         'model_file': 'wiki.en.tex.d100.model'
                        },
             'spanish': {'truth_file': 'summary-spanish-truth.txt',\
                         'model_file': 'wiki.es.tex.d100.model'
                        }
             }

tasks = ['age', 'gender']

num_features = 100
source = "wikipedia-self-trained"
poly_degrees = [1]
poly_C = [1]
rbf_gammas = [1]
rbf_C =[1]

all_results = {}


#for lang in relations.keys():

for lang in ["english"]:
    truth_file = relations[lang]['truth_file']
    model_file = current_working_dir + model_dir + relations[lang]['model_file']

    train = pd.read_csv(truth_file, header=0, delimiter="\t", quoting=1)
    print "Done reading file"
    clean_train_data = train['text']

    model = pickle.load( open( model_file, "rb" ) )

    trainDataVecs, trashedWords = helper.getAvgFeatureVecs( clean_train_data,\
                                                            model,\
                                                            num_features )

    print "Done making average vector"
    
    for task in tasks:
        train_y = train[task]
        poly_results = helper.doSVMwithPoly(trainDataVecs, train_y, source, \
                                            num_features, task, num_folds=10,\
                                            degrees=poly_degrees, C=poly_C)
        print "Done with polynomial", task
        rbf_results = helper.doSVMwithRBF(trainDataVecs, train_y, source, \
                                          num_features, task, num_folds=10,\
                                          gammas=rbf_gammas, C=rbf_C)

        print "Done with rbf", task

        results_one_task = helper.merge_two_dicts(poly_results, rbf_results)
        all_results = helper.merge_two_dicts(results_one_task, all_results)
