import os.path

import random
import math
import time
import sys
import string
import shlex, subprocess

from math import pow

import numpy as np

training_percentage = 0.7

seq_len = 50
peak_rate = 0.1
damp_rate = 0.3
max_feature_signal = 500
base = 3
perturbation_level = 0.2

def vec_to_str(v):
        s = ""
        for i in range(0, len(v)):
                s += str(v[i]) + " "

        s += '\n'
        return s

def vec_to_str_comma(v):
        s = ""
        for i in range(0, len(v)):
                s += str(v[i]) + ","
        
        s += '\n'
        return s

def write_data(arr, fn):
        f = open(fn, 'w')
        
        for i in range(0, len(arr)):
                f.write(vec_to_str(arr[i]))
        
        f.close()

def no_calc(x):
        return x

def sq_and_sqr(x):
        return (math.sqrt(max_feature_signal) * pow(x,0.5)) + (pow(x,2) / max_feature_signal)

def feature_peak_gen(s,max_num):
        return np.random.uniform(size=s) * max_num

def feature_peak_gen_exp(s,max_num):
        return np.power(base, get_rand(percentage=[0.7, 0.9, 0.97, 0.995], size=s))

def gen_data(s, max_num, nl, tot, weight_fun=no_calc, eval_fun=np.dot, gen_fun=feature_peak_gen):
        data = []
        weights = []
        
        for i in range(0, tot):
                #generate the feature peaks
                feature_peaks = gen_fun(s, max_num)
        	weight_peaks = (np.array(calc_weight_for_each(feature_peaks, weight_fun))) * peak_rate
        	target_peak = target_peak_gen(weight_peaks, feature_peaks, nl, eval_fun)
                
                output_peaks = np.append(feature_peaks, target_peak)
                data.append(output_peaks)
        	weights.append(weight_peaks)
        
        return (data, weights)

def calc_weight_for_each(peak_arr, wf):
        weight_arr = []
        
        for i in range(0, len(peak_arr)):
        	weight = 0
                multiplier = damp_rate
		#backward contribution
                for j in range(i-1, -1, -1):
			weight += wf(peak_arr[j]) * multiplier
			multiplier *= damp_rate

		#reset multiplier for forward
		multiplier = damp_rate
		for j in range(i+1, len(peak_arr), 1):
                        weight += wf(peak_arr[j]) * multiplier
                        multiplier *= damp_rate

		weight += peak_arr[i]

		weight_arr.append(weight)

	return weight_arr

def get_rand(percentage, size):
	gen_arr = []
        for i in range(0, size):
                var = np.random.uniform(-perturbation_level, perturbation_level)
                
		rn = np.random.random_sample()
                if rn < percentage[0]:
			val = 0
		elif rn < percentage[1]:
			val = 1
		elif rn < percentage[2]:
			val = 2
		elif rn < percentage[3]:
			val = 3
		else:
			val = 4
                                
                gen_arr.append(val + var)

	return np.array(gen_arr)

def target_peak_gen(r_weight, feature, nl, eval_fun):
        return eval_fun(r_weight, feature) * (1 + np.random.uniform(-nl, nl))

def str_to_float(arr):
	f_arr = []
        for i in range(0, len(arr)):
		f_arr.append(float(arr[i]))

	return np.array(f_arr)

def read_data(fn):
        data = []
        
	f = open(fn, 'r')
        for line in f:
		sl = line.split()
		digital_data = str_to_float(sl)
                data.append([digital_data[0:seq_len], digital_data[seq_len]])

	f.close()
	return data

def into_batchs(data, bs):
        batched_data = []
        single_batch_f = []
        single_batch_t = []
        
        
        for i in range(0, len(data)):
                single_batch_f.append(data[i][0])
                single_batch_t.append(data[i][1])
                
                if (i + 1) % bs == 0:
                        batched_data.append([np.array(single_batch_f), np.array(single_batch_t)])
                        single_batch_f = []
        		single_batch_t = []

	return batched_data

def write_with_list(data, write_list, fn):
        f = open(fn, 'w')
        
        for i in range(0, len(write_list)):
                c_data = data[write_list[i]]
                for j in range(0, len(c_data[1])):
			f.write(str(c_data[1][j]) + '\n')

	f.close()

def write_np_arr(data, fn):
	f = open(fn, 'w')

        for i in range(0, len(data)):
		f.write(vec_to_str(data[i]))

	f.close()

def read_pred(fn):
	f = open(fn, 'r')
        preds = []
        
        for line in f:
                sl = line.split()
		preds.append(float(sl[0]))
        
	f.close()
	return np.array(preds)

def calc_diff(ori, pred):
	diff = ori - pred
	return math.sqrt(np.dot(diff, diff) / np.dot(ori, ori))

def read_con(fn):
        f = open(fn, 'r')
        preds = []
        
        for line in f:
                sl = line.split()
                preds.append(np.array(str_to_float(sl)))

        f.close()
        return np.array(preds)

def cos_dist(a, b):
	return 1 - (np.dot(a, b) / math.sqrt(np.dot(a, a) * np.dot(b, b)))

def compare_con(ori, pred):
        dist_arr = []
        
        for i in range(0, len(ori)):
		dist_arr.append(cos_dist(ori[i], pred[i]))

	return np.array(dist_arr)

def write_csv(mean_arr, std_arr, noise_level):
	fn = "mean_and_std_" + str(noise_level) + ".csv"
	csv_f = open(fn, 'w')
        
        csv_f.write("sample mean \n")
        for i in range(0, len(mean_arr)):
		csv_f.write(vec_to_str_comma(mean_arr[i]))

	csv_f.write("sample std \n")
	for i in range(0, len(std_arr)):
        	csv_f.write(vec_to_str_comma(std_arr[i]))

	csv_f.close()



