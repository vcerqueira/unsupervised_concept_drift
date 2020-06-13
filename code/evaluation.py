#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math


def mtfa(detection_points, actual_d_point):
    ## mean time between false alarms
    ## how often we get alarms when there is no change
    dp = np.array(detection_points)

    dp_fa = dp[dp < actual_d_point]

    if len(dp_fa) == 1:
        mtfa_val = dp_fa[0]
    elif len(dp_fa) == 0:
        mtfa_val = math.nan 
    else:
        mtfa_val = np.nanmean(np.diff(dp_fa))
        
    return(mtfa_val)
    
    
def nfa(detection_points, actual_d_point):
    ## mean time between false alarms
    ## how often we get alarms when there is no change
    dp = np.array(detection_points)

    dp_fa = dp[dp < actual_d_point]

    return(len(dp_fa))
    
    
def mtd(detection_points, actual_d_point):
    ## mean time to detection
    
    dp = np.array(detection_points)
    dp_correct = dp[dp > actual_d_point]
    
    if len(dp_correct) > 0:
        first_succ_alarm = dp_correct[0]
        mtd_val = first_succ_alarm - actual_d_point
    else:
        mtd_val = math.nan
        
    return(mtd_val)
    
    
def missed_detection(detection_points, actual_d_point):
    ## misssed detection rate
    dp = np.array(detection_points)
    
    first_succ_alarm = dp[dp > actual_d_point]
    
    if len(first_succ_alarm) > 0:
        md_val = False
    else:
        md_val = True
    
    return(md_val)
       
def EvaluationMCapprx(mtfa_vals, mtd_vals, md_vals, nfa_vals):
    MDR = np.sum([int(x) for x in md_vals]) / len(md_vals)  
    MDR = np.round(MDR, 2)
    
    MTFA = np.round(np.nanmean(mtfa_vals), 1)
    MTD = np.round(np.nanmean(mtd_vals), 1)
    
    #MTFA_sdev = np.subtract(*np.nanpercentile(mtfa_vals, [75, 25]))
    MTFA_sdev = np.round(np.nanstd(mtfa_vals), 1)
    #MTD_sdev = np.subtract(*np.nanpercentile(mtd_vals, [75, 25]))
    MTD_sdev = np.round(np.nanstd(mtd_vals), 1)
    
    MTFA_sdev = np.round(MTFA_sdev,0)
    MTD_sdev = np.round(MTD_sdev,0)
    
    MTR = (MTFA / MTD) * (1-MDR)
    MTR = np.round(MTR, 2)
    
    #NFAR = np.sum([x>0 for x in nfa_vals]) / len(nfa_vals)
    
    r = dict({"MTFA": MTFA, 
              "MTD": MTD, 
              "MDR": MDR, 
              "MTR": MTR, 
              #"NFAR":NFAR,
              "MTFA_sdev": MTFA_sdev,
              "MTD_sdev": MTD_sdev})

    return(r)

def CycleAnalysis(detection_points,driftpoint):
    MTFA = mtfa(detection_points,driftpoint)
    MTD = mtd(detection_points,driftpoint)
    MD = missed_detection(detection_points,driftpoint)
    NFA = nfa(detection_points,driftpoint)
    
    result = dict({"MTFA":MTFA,"MTD":MTD,"MD":MD,"NFA":NFA})
    
    return(result)
    
