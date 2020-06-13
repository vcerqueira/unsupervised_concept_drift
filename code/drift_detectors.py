#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy

class HypothesisTestDetector(object):
    
    METHOD = "tt"
    def __init__(self, method, window, thr):
        assert method in ["ks","wrs","tt"]
        
        if method == "ks":
            #Two-sample Kolmogorov-Smirnov test
            m = scipy.stats.ks_2samp
        elif method == "wrs":
            #Wilcoxon rank-sum test
            m = scipy.stats.ranksums
        else:
            # Two-sample t-test
            m = scipy.stats.ttest_ind
        
        self.method = m
        self.alarm_list = []
        self.data = []
        self.window = window
        #self.get_change = False
        self.thr = thr
        self.index = 0
        
    def add_element(self, elem):
        self.data.append(elem)
        
    def detected_change(self):
        x = np.array(self.data)
        w = self.window
        
        if len(x) < 2*w:
            self.index += 1
            return(False)
            
        testw = x[-w:]
        refw = x[-(w*2):-w]
        
        ht = self.method(testw,refw)
        pval = ht[1]
        has_change = pval < self.thr
        
        if has_change:
            print('Change detected at index: ' + str(self.index))
            self.alarm_list.append(self.index)
            self.index += 1
            #self.get_change = True
            self.data = list(x[-w:])
            return(True)
        else: 
            self.index += 1
            return(False)
        
        
        
