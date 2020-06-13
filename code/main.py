#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import utils
from workflow import Workflow,Workflow_pt1,Workflow_pt2
from scipy.io import arff
import pickle

pd.set_option('display.max_columns', 50)


df = pd.read_csv("../data/electricity.csv")
df.rename(columns={"class": "target"}, inplace=True)
df['target'] = utils.col_as_int(df['target'].values)


y = df.target.values
X = df.drop(['target'], axis=1)

resultspt1 = Workflow_pt1(X,y,mcreps=50, inject_drift=True,
                   perc_train=0.6)

with open('results_elec_pt1.pkl', 'wb') as fp:
    pickle.dump(resultspt1, fp)


with open ('results_elec_pt1.pkl', 'rb') as fp:
    resultspt1 = pickle.load(fp)

resultspt2 = Workflow_pt2(resultspt1,window=500,
                   delta=0.001,
                   pval=0.001,
                   prob_instance=0.5, 
                   inst_delay=1000,
                   pht_thr=15)

with open('results_elec_pt2.pkl', 'wb') as fp:
    pickle.dump(resultspt2, fp)
    