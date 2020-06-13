#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from evaluation import EvaluationMCapprx
import pandas as pd
import numpy as np


l = 10
results = []
for i in range(1,l+1):
    print(i)
    #x = '../results/results_cov_pt2_2803_'+str(i)+'.pkl'
    x = '../results/results_elec_pt2_2503_'+str(i)+'.pkl'
    with open (x, 'rb') as fp:
        results.append(pickle.load(fp))
    

methods = list(results[0][0].keys())

resultsf = dict()
for m in methods:
    #m='S_ADWIN'
    MTFA_m = []
    MTD_m = []
    MD_m = []
    NFA_m = []
    NFAabs_m = []
    for i in range(l):
        #print(i)
        #i=9
        #len(results[i][0][m])
        r = results[i][0][m]
        r1 = results[i][1][m]
        #r[0][0]
        MTFA_m.append([x[0]['MTFA'] for x in r])
        MTD_m.append([x[0]['MTD'] for x in r])
        MD_m.append([x[0]['MD'] for x in r])
        NFA_m.append([x[0]['NFA'] for x in r])
        NFAabs_m.append(r1['ND'])
    
    MTFA_mf = [item for sublist in MTFA_m for item in sublist]
    MTD_mf = [item for sublist in MTD_m for item in sublist]
    MD_mf = [item for sublist in MD_m for item in sublist]
    NFA_mf = [item for sublist in NFA_m for item in sublist]
    ND_mf = np.round(np.mean(NFAabs_m),0)
    ND_std = np.round(np.std(NFAabs_m),0)
    
    resultsf[m] = EvaluationMCapprx(MTFA_mf,MTD_mf,MD_mf,NFA_mf)
    resultsf[m]["ND"] = ND_mf
    resultsf[m]["ND_sdev"] = ND_std
    

resf = pd.DataFrame(resultsf)
resf = resf.transpose()
resf = resf.rename(index={'WRS_Output': 'U_WRS_Output'})
resf = resf.rename(index={'TT_Output': 'U_TT_Output'})
resf = resf.rename(index={'KS_Output': 'U_KS_Output'})
resf = resf.rename(index={'WRS_Prob': 'UR_WRS_Prob'})
resf = resf.rename(index={'TT_Prob': 'UR_TT_Prob'})
resf = resf.rename(index={'KS_Prob': 'UR_KS_Prob'})

resf 
resf.to_csv("df_results_final_cov.csv")



    