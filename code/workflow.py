#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from evaluation import EvaluationMCapprx,CycleAnalysis
from drift_injection import Swapcols,Driftpoints
import copy
from skmultiflow.data.data_stream import DataStream
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import PageHinkley as PHT
from skmultiflow.meta import AdaptiveRandomForest as ARF
from skmultiflow.trees import RegressionHoeffdingTree as RHT
from drift_detectors import HypothesisTestDetector


def Workflow(X,y, mcreps, inject_drift, perc_train=0.6,window=100,delta=0.001,pval=0.01,prob_instance=0.1,inst_delay=1000):
    methods = ["S_ADWIN","S_PHT",
               "DS_ADWIN","DS_PHT",
               "WS_ADWIN","WS_PHT",
               "DWS_ADWIN","DWS_PHT",
               "U_ADWIN","U_PHT",
               "UR_ADWIN","UR_PHT",
               "WRS_Output","TT_Output","KS_Output",
               "WRS_Prob","TT_Prob","KS_Prob"]
    
    Results = []
    for i in range(mcreps):
        Ri = InnerCycle(X,y,
                        inject_drift=inject_drift,
                        perc_train = perc_train,
                        window=window,delta=delta,pval=pval,
                        prob_instance=prob_instance,inst_delay=inst_delay)
        Results.append(Ri)
    
    nmethods = len(Results[0][0])
    
    methodeval = dict()
    internal_results = dict()
    for j in range(nmethods):
        
        internal_results[methods[j]] = []
        MTFA_j = []
        MTD_j = []
        MD_j = []
        NFA_j = []
        CLR_j = []
        for i in range(mcreps):
            #i=0
            res = Results[i]
            driftpoint = res[1]["row"]
            cleanrun = res[1]["cleanrun"]
            alarmlist = res[0][j]
            
            r = CycleAnalysis(alarmlist,driftpoint)
            
            MTFA_j.append(r["MTFA"])
            MTD_j.append(r["MTD"])
            MD_j.append(r["MD"])
            NFA_j.append(r["NFA"])
            CLR_j.append(r["NFA"]/cleanrun)
            
            internal_results[methods[j]].append([r])
            
        evalres = EvaluationMCapprx(mtfa_vals=MTFA_j, 
                                    mtd_vals=MTD_j,
                                    md_vals=MD_j,
                                    nfa_vals=NFA_j)
        
        evalres["CLR"] = np.mean(CLR_j)
            
        methodeval[methods[j]] = evalres
            
    
    return([internal_results,methodeval])
 
    
    
def Workflow_pt1(X,y, mcreps, inject_drift, perc_train=0.6):
    
    Results = []
    for i in range(mcreps):
        Ri = InnerCycle_Train(X,y,
                        inject_drift=inject_drift,
                        perc_train = perc_train)
        Results.append(Ri)
    
    
    return(Results)
    
    

def Workflow_pt2(results_pt1,window=100,delta=0.001,pval=0.01,prob_instance=0.1,inst_delay=1000,pht_thr=15):
    #results_pt1 = [resultspt1[0]]
    
    methods = ["S_ADWIN","S_PHT",
               "DS_ADWIN","DS_PHT",
               "WS_ADWIN","WS_PHT",
               "DWS_ADWIN","DWS_PHT",
               "U_ADWIN","U_PHT",
               "UR_ADWIN","UR_PHT",
               "WRS_Output","TT_Output","KS_Output",
               "WRS_Prob","TT_Prob","KS_Prob"]
    
    mcreps = len(results_pt1)
    
    nmethods = len(methods)#len(Results[0][0])
    
    Results_2 = []
    for i in range(mcreps):
        Ri = InnerCycle_Apply(results_pt1[i],
                              window=window,
                              delta=delta,
                              pval=pval,
                              prob_instance=prob_instance,
                              inst_delay=inst_delay,
                              pht_thr=pht_thr)
        Results_2.append(Ri)
    
    
    Results = Results_2
    
    
    methodeval = dict()
    internal_results = dict()
    for j in range(nmethods):
        
        internal_results[methods[j]] = []
        MTFA_j = []
        MTD_j = []
        MD_j = []
        NFA_j = []
        CLR_j = []
        for i in range(mcreps):
            #i=0
            res = Results[i]
            #res = Results[0]
            
            
            driftpoint = res[1]["row"]
            cleanrun = res[1]["cleanrun"]
            alarmlist = res[0][j]
            
            r = CycleAnalysis(alarmlist,driftpoint)
            
            MTFA_j.append(r["MTFA"])
            MTD_j.append(r["MTD"])
            MD_j.append(r["MD"])
            NFA_j.append(r["NFA"])
            CLR_j.append(r["NFA"]/cleanrun)
            
            internal_results[methods[j]].append([r])
            
        evalres = EvaluationMCapprx(mtfa_vals=MTFA_j, 
                                    mtd_vals=MTD_j,
                                    md_vals=MD_j,
                                    nfa_vals=NFA_j)
        
        evalres["CLR"] = np.mean(CLR_j)
        evalres["ND"] = len(alarmlist)
            
        methodeval[methods[j]] = evalres
            
    
    return([internal_results,methodeval])
     
     

def InnerCycle(X,y,inject_drift,perc_train,window,delta,pval,prob_instance, inst_delay):
    
    # get number of training samples
    ntrain = int(perc_train * X.shape[0])
    
    if inject_drift:
        # pick a point between 0.7 and 0.9 of the stream
        dpoints = Driftpoints(X)
        dpoints["cleanrun"] = dpoints["row"] - ntrain
        
        # contaminate X after that point
        X = Swapcols(df=X,
                     class_vec=y, 
                     ids=dpoints["cols"], 
                     t_change=dpoints["row"])
    else:
        dpoints = dict({"row":X.shape[0],"cols":0})
    
    
    
    # cast data as DataStream class
    stream = DataStream(X, y)
    stream.prepare_for_use()
    # call incr model (main classifier, teacher model)
    stream_clf = ARF(n_estimators=25, 
                     drift_detection_method=None, 
                     warning_detection_method=None)
    
    
    # get training data... first ntrain rows
    Xtrain,ytrain = stream.next_sample(ntrain)
    
    # partial fit of the incre model using training data
    stream_clf.fit(Xtrain,ytrain,classes=stream.target_values)
    yhat_train = stream_clf.predict(Xtrain)
    yhat_train_prob = stream_clf.predict_proba(Xtrain)### needs warnings!!!!!!!!!
    yhat_tr_max_prob = np.array([np.max(x) for x in yhat_train_prob])
    
    # fit student model
    student_clf = ARF(n_estimators=25, 
                      drift_detection_method=None, 
                      warning_detection_method=None)
    student_clf.fit(Xtrain,yhat_train,classes=stream.target_values)
    
    student_regr = RHT()
    student_regr.fit(Xtrain,yhat_tr_max_prob)
    
    ####### Call drift detectors
    
    ## Supervised
    # Supervised with ADWIN
    S_ADWIN = ADWIN()#(delta=delta)
    S_ADWIN_alarms = [] 
    # Supervised with PHT
    S_PHT = PHT()#(min_instances=window,delta=delta)
    S_PHT_alarms = []
    # Delayed Supervised with ADWIN
    DS_ADWIN = ADWIN()#(delta=delta)
    DS_ADWIN_alarms = [] 
    # Delayed Supervised with PHT
    DS_PHT = PHT()#(min_instances=window,delta=delta)
    DS_PHT_alarms = []
    
    ## Semi-supervised
    # Semi-Supervised with ADWIN
    WS_ADWIN = ADWIN()#(delta=delta)
    WS_ADWIN_alarms = [] 
    # Supervised with PHT
    WS_PHT = PHT()#(min_instances=window,delta=delta)
    WS_PHT_alarms = []
    # Delayed Supervised with ADWIN
    DWS_ADWIN = ADWIN()#(delta=delta)
    DWS_ADWIN_alarms = [] 
    # Delayed Supervised with PHT
    DWS_PHT = PHT()#(min_instances=window,delta=delta)
    DWS_PHT_alarms = []
    
    
    ##### Unsupervised
    # Student with ADWIN
    U_ADWIN = ADWIN()#(delta=delta)
    U_ADWIN_alarms = []
    # Student with PHT
    U_PHT = PHT()#(min_instances=window,delta=delta)
    U_PHT_alarms = []
    
    # Student with ADWIN
    UR_ADWIN = ADWIN()#(delta=delta)
    UR_ADWIN_alarms = []
    # Student with PHT
    UR_PHT = PHT()#(min_instances=window,delta=delta)
    UR_PHT_alarms = []
    
    # WRS with output
    WRS_Output = HypothesisTestDetector(method="wrs",window=window,thr=pval)
    WRS_Output_alarms = []
    # WRS with class prob
    WRS_Prob = HypothesisTestDetector(method="wrs",window=window,thr=pval)
    WRS_Prob_alarms = []
    # TT with output
    TT_Output = HypothesisTestDetector(method="tt",window=window,thr=pval)
    TT_Output_alarms = []
    # TT with class prob
    TT_Prob = HypothesisTestDetector(method="tt",window=window,thr=pval)
    TT_Prob_alarms = []
    # KS with output
    KS_Output = HypothesisTestDetector(method="ks",window=window,thr=pval)
    KS_Output_alarms = []
    # KS with class prob
    KS_Prob = HypothesisTestDetector(method="ks",window=window,thr=pval)
    KS_Prob_alarms = []
    
    Driftmodels = [S_ADWIN,S_PHT,
                   DS_ADWIN,DS_PHT,
                   WS_ADWIN,WS_PHT,
                   DWS_ADWIN,DWS_PHT,
                   U_ADWIN,U_PHT,
                   UR_ADWIN,UR_PHT,
                   WRS_Output,TT_Output,KS_Output,
                   WRS_Prob,TT_Prob,KS_Prob]
    
    Driftmodels_alarms = [S_ADWIN_alarms,S_PHT_alarms,
                          DS_ADWIN_alarms,DS_PHT_alarms,
                          WS_ADWIN_alarms,WS_PHT_alarms,
                          DWS_ADWIN_alarms,DWS_PHT_alarms,
                          U_ADWIN_alarms,U_PHT_alarms,
                          UR_ADWIN_alarms,UR_PHT_alarms,
                          WRS_Output_alarms,TT_Output_alarms,KS_Output_alarms,
                          WRS_Prob_alarms,TT_Prob_alarms,KS_Prob_alarms]
    
    
    S_driftmodels = Driftmodels[0:2]
    DS_driftmodels = Driftmodels[2:4]
    WS_driftmodels = Driftmodels[4:6]
    DWS_driftmodels = Driftmodels[6:8]
    Ustd_driftmodels = Driftmodels[8:10]
    Ustdreg_driftmodels = Driftmodels[10:12]
    Uoutput_driftmodels = Driftmodels[12:15]
    Uprob_driftmodels = Driftmodels[15:18]
    
    # always updated
    S_clf = copy.deepcopy(stream_clf)
    # always updated with delay
    DS_clf = copy.deepcopy(stream_clf)
    # updated immediately with some prob
    WS_clf = copy.deepcopy(stream_clf)
    # updated with delay with some prob
    DWS_clf = copy.deepcopy(stream_clf)
    # never updated
    U_clf = copy.deepcopy(stream_clf)
    
    i=ntrain
    k=0
    DWS_yhat_hist = []
    DS_yhat_hist = []
    X_hist = []
    y_hist = []
    while(stream.has_more_samples()):
        print(i)
        #i=3000
        Xi,yi = stream.next_sample()
        
        y_hist.append(yi[0])
        X_hist.append(Xi)
        
        ext_Xi = np.concatenate([Xtrain[-10:],Xi])

        U_prob = U_clf.predict_proba(ext_Xi)[-1]        
        U_yhat = U_clf.predict(ext_Xi)[-1]
        S_yhat = S_clf.predict(ext_Xi)[-1]
        WS_yhat = WS_clf.predict(ext_Xi)[-1]
        DS_yhat = DS_clf.predict(ext_Xi)[-1]
        DWS_yhat = DWS_clf.predict(ext_Xi)[-1]
        
        DWS_yhat_hist.append(DWS_yhat)
        DS_yhat_hist.append(DS_yhat)
        
        if len(U_prob) < 2:
            U_yhat_prob_i = U_prob[0]
        elif len(U_prob) == 2:
            U_yhat_prob_i = U_prob[1]
        else:
            U_yhat_prob_i = np.max(U_prob)
        
        y_meta_hat_i = student_clf.predict(ext_Xi)[-1]
        y_meta_prob = student_regr.predict(ext_Xi)[-1]
        
        # Updating student model
        student_clf.partial_fit(Xi,[U_yhat])
        # Updating supervised model 
        S_clf.partial_fit(Xi,yi)
        
        
        # Computing loss
        S_err_i = int(yi[0] != S_yhat)
        student_err_i = int(y_meta_hat_i != U_yhat)
        student_prob_err_i = U_yhat_prob_i - y_meta_prob
        
        for model in S_driftmodels:
            model.add_element(S_err_i)
            
        for model in Ustd_driftmodels:
            model.add_element(student_err_i)
            
        for model in Ustdreg_driftmodels:
            model.add_element(student_prob_err_i)
        
        for model in Uoutput_driftmodels:
            model.add_element(U_yhat)
        
        for model in Uprob_driftmodels:
            model.add_element(U_yhat_prob_i)
            
        put_i_available = np.random.binomial(1,prob_instance)
        
        if k >= inst_delay:
            DS_err_i = int(y_hist[k-inst_delay] != DS_yhat_hist[k-inst_delay])
            DS_clf.partial_fit(X_hist[k-inst_delay],[y_hist[k-inst_delay]])
            for model in DS_driftmodels:
                model.add_element(DS_err_i)
                
            if put_i_available > 0:
                DWS_err_i = int(y_hist[k-inst_delay] != DWS_yhat_hist[k-inst_delay])
                DWS_clf.partial_fit(X_hist[k-inst_delay],[y_hist[k-inst_delay]])
                for model in DWS_driftmodels:
                    model.add_element(DWS_err_i)
        
        if put_i_available > 0:
            WS_err_i = int(yi[0] != WS_yhat)
            WS_clf.partial_fit(Xi,yi)
            for model in WS_driftmodels:
                model.add_element(WS_err_i)
            
        # detect changes
        for j,model in enumerate(Driftmodels):
            has_change = model.detected_change()
            if has_change:
                Driftmodels_alarms[j].append(i)
                
        i+=1
        k+=1

        
    return([Driftmodels_alarms,dpoints])
  
    


def InnerCycle_Train(X,y,inject_drift,perc_train):
    
    # get number of training samples
    ntrain = int(perc_train * X.shape[0])
    
    if inject_drift:
        # pick a point between 0.7 and 0.9 of the stream
        dpoints = Driftpoints(X)
        dpoints["cleanrun"] = dpoints["row"] - ntrain
        
        # contaminate X after that point
        X = Swapcols(df=X,
                     class_vec=y, 
                     ids=dpoints["cols"], 
                     t_change=dpoints["row"])
    else:
        dpoints = dict({"row":X.shape[0],"cols":0})
    
    
    
    # cast data as DataStream class
    stream = DataStream(X, y)
    stream.prepare_for_use()
    # call incr model (main classifier, teacher model)
    stream_clf = ARF(n_estimators=25)#, 
                     #drift_detection_method=None, 
                     #warning_detection_method=None
                     #)
    
    
    # get training data... first ntrain rows
    Xtrain,ytrain = stream.next_sample(ntrain)
    
    # partial fit of the incre model using training data
    stream_clf.fit(Xtrain,ytrain,classes=stream.target_values)
    yhat_train = stream_clf.predict(Xtrain)
    yhat_train_prob = stream_clf.predict_proba(Xtrain)### needs warnings!!!!!!!!!
    yhat_tr_max_prob = np.array([np.max(x) for x in yhat_train_prob])
    
    # fit student model
    student_clf = ARF(n_estimators=25)#, 
                      #drift_detection_method=None, 
                      #warning_detection_method=None)
    student_clf.fit(Xtrain,yhat_train,classes=stream.target_values)
    
    student_regr = RHT()
    student_regr.fit(Xtrain,yhat_tr_max_prob)
    
    results = dict()
    results["Teacher"] = stream_clf 
    results["Student"] = student_clf
    results["StudentRegression"] = student_regr
    results["Driftpoints"] = dpoints
    results["n"] = ntrain
    results["Stream"] = stream
    results["Xtrain"] = Xtrain
    
        
    return(results)




def InnerCycle_Apply(trainResults,window,delta,pval,prob_instance, inst_delay,pht_thr):
    #trainResults=results_pt1[0]
    
    
    stream_clf = trainResults["Teacher"] 
    student_clf = trainResults["Student"]
    student_regr = trainResults["StudentRegression"]
    dpoints = trainResults["Driftpoints"]
    ntrain = trainResults["n"]
    stream = trainResults["Stream"]
    Xtrain = trainResults["Xtrain"]
    
    ####### Call drift detectors
    
    #pht_thr = 15
    ## Supervised
    # Supervised with ADWIN
    S_ADWIN = ADWIN()#(delta=delta)
    S_ADWIN_alarms = [] 
    # Supervised with PHT
    S_PHT = PHT(threshold=pht_thr)#(min_instances=window,delta=delta)
    S_PHT_alarms = []
    # Delayed Supervised with ADWIN
    DS_ADWIN = ADWIN()#(delta=delta)
    DS_ADWIN_alarms = [] 
    # Delayed Supervised with PHT
    DS_PHT = PHT(threshold=pht_thr)#(min_instances=window,delta=delta)
    DS_PHT_alarms = []
    
    ## Semi-supervised
    # Semi-Supervised with ADWIN
    WS_ADWIN = ADWIN()#(delta=delta)
    WS_ADWIN_alarms = [] 
    # Supervised with PHT
    WS_PHT = PHT(threshold=pht_thr)#(min_instances=window,delta=delta)
    WS_PHT_alarms = []
    # Delayed Supervised with ADWIN
    DWS_ADWIN = ADWIN()#(delta=delta)
    DWS_ADWIN_alarms = [] 
    # Delayed Supervised with PHT
    DWS_PHT = PHT(threshold=pht_thr)#(min_instances=window,delta=delta)
    DWS_PHT_alarms = []
    
    
    ##### Unsupervised
    # Student with ADWIN
    U_ADWIN = ADWIN()#(delta=delta)
    U_ADWIN_alarms = []
    # Student with PHT
    U_PHT = PHT(threshold=pht_thr)#(min_instances=window,delta=delta)
    U_PHT_alarms = []
    
    # Student with ADWIN
    UR_ADWIN = ADWIN()#(delta=delta)
    UR_ADWIN_alarms = []
    # Student with PHT
    UR_PHT = PHT(threshold=pht_thr)#(min_instances=window,delta=delta)
    UR_PHT_alarms = []
    
    # WRS with output
    WRS_Output = HypothesisTestDetector(method="wrs",window=window,thr=pval)
    WRS_Output_alarms = []
    # WRS with class prob
    WRS_Prob = HypothesisTestDetector(method="wrs",window=window,thr=pval)
    WRS_Prob_alarms = []
    # TT with output
    TT_Output = HypothesisTestDetector(method="tt",window=window,thr=pval)
    TT_Output_alarms = []
    # TT with class prob
    TT_Prob = HypothesisTestDetector(method="tt",window=window,thr=pval)
    TT_Prob_alarms = []
    # KS with output
    KS_Output = HypothesisTestDetector(method="ks",window=window,thr=pval)
    KS_Output_alarms = []
    # KS with class prob
    KS_Prob = HypothesisTestDetector(method="ks",window=window,thr=pval)
    KS_Prob_alarms = []
    
    Driftmodels = [S_ADWIN,S_PHT,
                   DS_ADWIN,DS_PHT,
                   WS_ADWIN,WS_PHT,
                   DWS_ADWIN,DWS_PHT,
                   U_ADWIN,U_PHT,
                   UR_ADWIN,UR_PHT,
                   WRS_Output,TT_Output,KS_Output,
                   WRS_Prob,TT_Prob,KS_Prob]
    
    Driftmodels_alarms = [S_ADWIN_alarms,S_PHT_alarms,
                          DS_ADWIN_alarms,DS_PHT_alarms,
                          WS_ADWIN_alarms,WS_PHT_alarms,
                          DWS_ADWIN_alarms,DWS_PHT_alarms,
                          U_ADWIN_alarms,U_PHT_alarms,
                          UR_ADWIN_alarms,UR_PHT_alarms,
                          WRS_Output_alarms,TT_Output_alarms,KS_Output_alarms,
                          WRS_Prob_alarms,TT_Prob_alarms,KS_Prob_alarms]
    
    
    S_driftmodels = Driftmodels[0:2]
    DS_driftmodels = Driftmodels[2:4]
    WS_driftmodels = Driftmodels[4:6]
    DWS_driftmodels = Driftmodels[6:8]
    Ustd_driftmodels = Driftmodels[8:10]
    Ustdreg_driftmodels = Driftmodels[10:12]
    Uoutput_driftmodels = Driftmodels[12:15]
    Uprob_driftmodels = Driftmodels[15:18]
    
    # always updated
    S_clf = copy.deepcopy(stream_clf)
    # always updated with delay
    DS_clf = copy.deepcopy(stream_clf)
    # updated immediately with some prob
    WS_clf = copy.deepcopy(stream_clf)
    # updated with delay with some prob
    DWS_clf = copy.deepcopy(stream_clf)
    # never updated
    U_clf = copy.deepcopy(stream_clf)
    
    i=ntrain
    k=0
    DWS_yhat_hist = []
    DS_yhat_hist = []
    X_hist = []
    y_hist = []
    while(stream.has_more_samples()):
        print(i)
        #i=3000
        Xi,yi = stream.next_sample()
        
        y_hist.append(yi[0])
        X_hist.append(Xi)
        
        ext_Xi = np.concatenate([Xtrain[-10:],Xi])

        U_prob = U_clf.predict_proba(ext_Xi)[-1]        
        U_yhat = U_clf.predict(ext_Xi)[-1]
        S_yhat = S_clf.predict(ext_Xi)[-1]
        WS_yhat = WS_clf.predict(ext_Xi)[-1]
        DS_yhat = DS_clf.predict(ext_Xi)[-1]
        DWS_yhat = DWS_clf.predict(ext_Xi)[-1]
        
        DWS_yhat_hist.append(DWS_yhat)
        DS_yhat_hist.append(DS_yhat)
        
        if len(U_prob) < 2:
            U_yhat_prob_i = U_prob[0]
        elif len(U_prob) == 2:
            U_yhat_prob_i = U_prob[1]
        else:
            U_yhat_prob_i = np.max(U_prob)
        
        y_meta_hat_i = student_clf.predict(ext_Xi)[-1]
        y_meta_prob = student_regr.predict(ext_Xi)[-1]
        
        # Updating student model
        student_clf.partial_fit(Xi,[U_yhat])
        # Updating supervised model 
        S_clf.partial_fit(Xi,yi)
        
        
        # Computing loss
        S_err_i = int(yi[0] != S_yhat)
        student_err_i = int(y_meta_hat_i != U_yhat)
        student_prob_err_i = U_yhat_prob_i - y_meta_prob
        
        for model in S_driftmodels:
            model.add_element(S_err_i)
            
        for model in Ustd_driftmodels:
            model.add_element(student_err_i)
            
        for model in Ustdreg_driftmodels:
            model.add_element(student_prob_err_i)
        
        for model in Uoutput_driftmodels:
            model.add_element(U_yhat)
        
        for model in Uprob_driftmodels:
            model.add_element(U_yhat_prob_i)
            
        put_i_available = np.random.binomial(1,prob_instance)
        
        if k >= inst_delay:
            DS_err_i = int(y_hist[k-inst_delay] != DS_yhat_hist[k-inst_delay])
            DS_clf.partial_fit(X_hist[k-inst_delay],[y_hist[k-inst_delay]])
            for model in DS_driftmodels:
                model.add_element(DS_err_i)
                
            if put_i_available > 0:
                DWS_err_i = int(y_hist[k-inst_delay] != DWS_yhat_hist[k-inst_delay])
                DWS_clf.partial_fit(X_hist[k-inst_delay],[y_hist[k-inst_delay]])
                for model in DWS_driftmodels:
                    model.add_element(DWS_err_i)
        
        if put_i_available > 0:
            WS_err_i = int(yi[0] != WS_yhat)
            WS_clf.partial_fit(Xi,yi)
            for model in WS_driftmodels:
                model.add_element(WS_err_i)
            
        # detect changes
        for j,model in enumerate(Driftmodels):
            has_change = model.detected_change()
            if has_change:
                Driftmodels_alarms[j].append(i)
                
        i+=1
        k+=1

        
    return([Driftmodels_alarms,dpoints])



