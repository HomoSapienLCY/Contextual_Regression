import os.path

import random
import math
import time
import sys
import string
import shlex, subprocess

import numpy as np
import scipy.stats

folder = sys.argv[1]
orname_base = folder + "/original_peaks_"
prename_base = folder + "/predicted_peaks_"

start = 0
end = 300
interval = 15

def peak_reading(fn):
	peaks = []
	f = open(fn, 'r')

        for line in f:
		sl = line.split()
		peaks.append(float(sl[1]))

	f.close()
	return peaks

def one_percent_ov(ori_top_ind, b):
        one_percent_num = len(b) / 100
        
	b_sort_ind = sorted(range(len(b)), key=lambda k: b[k], reverse=True)[1:one_percent_num]

	common = set(ori_top_ind) & set(b_sort_ind)

	return float(len(common)) / float(one_percent_num)

def five_percent_ov(ori_top_ind, b):
        one_percent_num = len(b) / 100
        
        b_sort_ind = sorted(range(len(b)), key=lambda k: b[k], reverse=True)[1:(one_percent_num * 5)]
        
        common = set(ori_top_ind) & set(b_sort_ind)
        
        return float(len(common)) / float(one_percent_num)

#read_originals
ori_train = peak_reading(orname_base + "train")
ori_test = peak_reading(orname_base + "test")

ori_train_top_ind_1 = sorted(range(len(ori_train)), key=lambda k: ori_train[k], reverse=True)[1:(len(ori_train)/100)]
ori_test_top_ind_1 = sorted(range(len(ori_test)), key=lambda k: ori_test[k], reverse=True)[1:(len(ori_test)/100)]

ori_train_top_ind_5 = sorted(range(len(ori_train)), key=lambda k: ori_train[k], reverse=True)[1:(len(ori_train)/20)]
ori_test_top_ind_5 = sorted(range(len(ori_test)), key=lambda k: ori_test[k], reverse=True)[1:(len(ori_test)/20)]

for i in range(start, end + 1, interval):
        if i == end:
        	tail = "final"
        else:
        	tail = str(i)
        
	ptr = peak_reading(prename_base + "train_" + tail)
	pte = peak_reading(prename_base + "test_" + tail)

	#Calculate pearson
	trcorr, trpv = scipy.stats.pearsonr(ori_train, ptr)
        tecorr, tepv = scipy.stats.pearsonr(ori_test, pte)

	print "corr train at " + tail + ": "
        print trcorr
        print "corr test at " + tail + ": "
        print tecorr

	#Calculate match 1
	trone = one_percent_ov(ori_train_top_ind_1, ptr)
	teone = one_percent_ov(ori_test_top_ind_1, pte)

	print "train 1% at " + tail + ": "
	print trone
	print "test 1% at " + tail + ": "
	print teone

	#Calculate Cap1Obs
	trone_obs = five_percent_ov(ori_train_top_ind_1, ptr)
	teone_obs = five_percent_ov(ori_test_top_ind_1, pte)

	print "train 1% obs at " + tail + ": "
        print trone_obs
        print "test 1% obs at " + tail + ": "
        print teone_obs

	#Calculate Cap1Imp
	trone_imp = one_percent_ov(ori_train_top_ind_5, ptr)
        teone_imp = one_percent_ov(ori_test_top_ind_5, pte)

	print "train 1% imp at " + tail + ": "
        print trone_imp
        print "test 1% imp at " + tail + ": "
        print teone_imp














