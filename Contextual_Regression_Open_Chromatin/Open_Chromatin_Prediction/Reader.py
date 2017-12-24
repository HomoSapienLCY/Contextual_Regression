import os.path

import random
import math
import time
import sys
import string
import shlex, subprocess

import numpy as np
import pyBigWig as bw

def str_to_bool(s):
        if s == "True":
                return True
        elif s == "False":
                return False
	else:
        	raise ValueError

#read big wig
def readBigWig(filename, chrom, binSize):
	fl = bw.open(filename)
	length = fl.chroms(chrom)
	result = fl.stats(chrom, 0, length, nBins=length // binSize, exact=True)
	#remove all the Nones
	filtered_result = [(ele if ele != None else 0) for ele in result]
	fl.close()
	return filtered_result

#read big wig and log
def readBigWig_log(filename, chrom, binSize):
        fl = bw.open(filename)
        length = fl.chroms(chrom)
        result = fl.stats(chrom, 0, length, nBins=length // binSize, exact=True)
        #remove all the Nones
        filtered_result = [math.log((ele if ele != None else 0) + 1) for ele in result]
        fl.close()
        return filtered_result

#read big wig with max
def readBigWigMax(filename, chrom, binSize):
        fl = bw.open(filename)
        length = fl.chroms(chrom)
        result = fl.stats(chrom, 0, length, nBins=length // binSize, type="max")
        #remove all the Nones
        filtered_result = [(ele if ele != None else 0) for ele in result]
        fl.close()
        return filtered_result

#read big wig with max and log
def readBigWigMax_log(filename, chrom, binSize):
        fl = bw.open(filename)
        length = fl.chroms(chrom)
        result = fl.stats(chrom, 0, length, nBins=length // binSize, type="max")
        #remove all the Nones
        filtered_result = [math.log((ele if ele != None else 0) + 1) for ele in result]
        fl.close()
        return filtered_result

def unnormalized(datum, mean, std):
        return (datum * std) + mean

def assign_one_hot(cha):
        if cha == 'A':
                return [1, 0, 0, 0]
        elif cha == 'C':
                return [0, 1, 0, 0]
        elif cha == 'G':
                return [0, 0, 1, 0]
	elif cha == 'T':
        	return [0, 0, 0, 1]
        elif cha == 'N':
                return [0.25, 0.25, 0.25, 0.25]
	else:
        	print "one hot assignment runs into a problem"
                return []

#read seq file
def read_seq(filename):
        seq_file = open(filename, 'r')
        seq_data = []
        
        for line in seq_file:
                seq_pieces = list(line.strip().upper())
                if(seq_pieces[0] != '>'):
                        seq_data.extend(seq_pieces)
        
	seq_file.close()
        
        return ''.join(seq_data)

#convert sequence to one hot matrix
def seq_to_onehot(seq):
        return np.array([assign_one_hot(seq[i]) for i in range(0, len(seq))])

def top_percent_array_index(data, percent):
        sorted_data = sorted(data)
        data_len = len(sorted_data)
        percent_position = int(math.ceil(float(data_len) * float(1 - percent)))
        percent_cut = sorted_data[percent_position]
        
        return [i for i in range (0,data_len) if (data[i] >= percent_cut)]

def weighted_sampling(top_percent_index_array, bottom_percent_index_array, top_percent_sampling_rate):
        rn = random.uniform(0, 1)
        #sample from top
        if(rn <= top_percent_sampling_rate):
                ri = random.randint(0,len(top_percent_index_array) - 1)
                return top_percent_index_array[ri]
        #sample from bottom
        else:
                ri = random.randint(0,len(bottom_percent_index_array) - 1)
                return bottom_percent_index_array[ri]

def write_peaks(filename, data, offset):
        pf = open(filename, 'a')
        for i in range(0, len(data)):
                pf.write(str(i + offset) + ' ' + str(data[i]) + '\n')
        pf.close()

def log_the_array(array):
        result = []
        for ele in array:
                result.append(math.log(ele))
        return result

def cal_mean(data):
        sum = 0
        for datum in data:
                sum += datum
        
        return float(sum) / float(len(data))

def cal_std(data):
        mean = cal_mean(data)
        square_sum = 0
        for datum in data:
                square_sum += (datum - mean) * (datum - mean)
        
        var = float(square_sum) / float(len(data))
        return math.sqrt(var)

def normalize(data):
        mean = cal_mean(data)
        std = cal_std(data)
        data_normalized = []
        for i in range(0, len(data)):
                current_normalized = float(data[i] - mean) / float(std)
                data_normalized.append(current_normalized)
        
        return np.array(data_normalized)

#flatten the outlier data to be as high as a normal one
def flattening(data, tolerance):
        copy = []
        for i in range(0, len(data)):
                if(data[i] > tolerance):
                        copy.append(tolerance)
                else:
                        copy.append(data[i])
	return np.array(copy)

def transform_into_p(data):
        one_minus_p_values = []
        for datum in data:
                one_minus_p_values.append(1 - math.pow(0.1,datum))
        return np.array(one_minus_p_values)

#extend the blacklist region to include the nearest block
def blacklist_blockizer(blacklist_dictionary, block):
        after_block_dict = {}
        blank_size_dict = {}
        
        for chromo in blacklist_dictionary.keys():
                counter = 0
                append_regions = []
                
                regions = blacklist_dictionary[chromo]
                
                for region in regions:
                        up_block = region[0] // block
                        down_block = (region[1] // block) + 1
                        
                        append_regions.append([up_block * block, down_block * block])
                        counter += down_block - up_block
                
                after_block_dict[chromo] = append_regions
                blank_size_dict[chromo] = counter
        
        return [after_block_dict, blank_size_dict]

#generate a list of learnable training indeces from both tpi and bpi
#put the rest into test set
#sample the training and testing, return 2 arrays of index
def sample_from_sets(whole_set, percentage):
        total_size = len(whole_set)
        
        train_size = int(total_size * percentage)
        test_size = total_size - train_size
        
        training_set = random.sample(whole_set, train_size)
        
        return training_set

def sample_sets(whole_set, percentage):
        training_set = sample_from_sets(whole_set, percentage)
        test_set = list(set(whole_set) - set(training_set))
        
        return training_set, test_set

def write_list_file(array, fname):
        list_file = open(fname, 'w')
        list_file.write("start  end \n")
        
        for ele in array:
                list_file.write(str(ele * chunk) + "  " + str((ele + 1) * chunk - 1) + " \n")
        
        list_file.close()

def write_list_for_retrieval(array, fname):
        list_file = open(fname, 'w')
        for ele in array:
                list_file.write(str(ele) + " ")
        
        list_file.close()

def str_to_int(array):
        t = []
        for ele in array:
                t.append(int(ele))
        
        return t

def get_list_from_file(fname):
        return str_to_int(open(fname, 'r').readline().split())

#get data
def obtain_data_from_list(the_data, list):
        obtained_data = []
        for i in list:
                obtained_data.append(the_data[i])
        
        return np.array(obtained_data)

#obtain from list, pack data and clear
def data_packing(the_data, train_list, test_list):
        train = obtain_data_from_list(the_data, train_list)
        test = obtain_data_from_list(the_data, test_list)
        
        shaped_train = np.reshape(train.reshape(-1), (-1, batch_size, seq_len))
        shaped_test = np.reshape(test.reshape(-1), (-1, batch_size, seq_len))
        
        return [shaped_train, shaped_test]

def seq_data_packing(the_seq_data, train_list, test_list):
        train = obtain_data_from_list(the_seq_data, train_list)
        test = obtain_data_from_list(the_seq_data, test_list)
        
        shaped_train = np.reshape(train.reshape(-1), (-1, batch_size, seq_len, resolution, base_type_size))
        shaped_test = np.reshape(test.reshape(-1), (-1, batch_size, seq_len, resolution, base_type_size))
        
        return [shaped_train, shaped_test]

#complete procedure of blacklist, reshape and pack
def read_black_reshape_pack(filename, chrom_sp, train_list, test_list):
        d_x = np.array(readBigWig_log(filename, chrom_sp, resolution))
        d_x = np.delete(d_x, bl)
        d_x = np.reshape(d_x[:n], (-1, seq_len))
        d_x = data_packing(d_x, train_list, test_list)
        
        print("finish packing " + filename)
        
        return d_x

#read the sequence, remove the region
def read_black_reshape_pack_sequence(chrom_sp, train_list, test_list):
        d_x_seq = read_seq(mark_folder + chrom_sp + ".fa")
        #cut the tail
        d_x_seq = d_x_seq[0:len(d_x_seq)//resolution*resolution]
        #reshape into resolution size segments
        d_x_seq = [d_x_seq[i:i+resolution] for i in range(0, len(d_x_seq), resolution)]
        #remove bl
        d_x_seq = np.array([seq_to_onehot(d_x_seq[i]) for i in range(0, len(d_x_seq)) if i not in bl])
        #cut into sequence length
        d_x_seq = np.reshape(d_x_seq[:n], (-1, seq_len, resolution, base_type_size))
        #pack sequence
        d_x_seq = seq_data_packing(d_x_seq, train_list, test_list)
        
        print("finish packing " + chrom_sp + ".fa")
        
        return d_x_seq

def pack_concatenate(pack_array):
        train = []
        test = []
        
        for i in range(0, len(pack_array)):
                train.append(np.expand_dims(pack_array[i][0], axis = 3))
                test.append(np.expand_dims(pack_array[i][1], axis = 3))

        train_pack = np.concatenate(tuple(train), axis = 3)
        test_pack = np.concatenate(tuple(test), axis = 3)

	return [train_pack, test_pack]

def combine_box(data_box):
        train = []
        test = []
        
        for i in range(0, len(data_box)):
                train.append(data_box[i][0])
                test.append(data_box[i][1])
        
        train_box = np.concatenate(tuple(train), axis = 0)
        test_box = np.concatenate(tuple(test), axis = 0)

	return [train_box, test_box]

def write_con_pred(v_con_data, h_con_data, prediction_data, v_spacing, h_spacing, pred_spacing, mark):
        write_peaks("v_context_" + mark, v_con_data.reshape(-1), v_spacing)
        write_peaks("h_context_" + mark, h_con_data.reshape(-1), h_spacing)
        write_peaks("predicted_peaks_" + mark, prediction_data.reshape(-1), pred_spacing)

def write_ms_con_pred(ms_data, context_data, prediction_data, ms_spacing, con_spacing, pred_spacing, mark):
        write_peaks("marksum_" + mark, ms_data.reshape(-1), ms_spacing)
        write_peaks("context_" + mark, context_data.reshape(-1), con_spacing)
        write_peaks("predicted_peaks_" + mark, prediction_data.reshape(-1), pred_spacing)

#obtain black
def obtain_black(chrom, resolution, bbd):
        chrom_black_list = bbd[chrom]
        
        blacklist_cu = []
        for intervel in chrom_black_list:
                blacklist_cu.extend(range(intervel[0], intervel[1], resolution))
        
        for i in range(0, len(blacklist_cu)):
                blacklist_cu[i] = int(blacklist_cu[i] / resolution)
        
        return blacklist_cu

def interval_calc(arr, total):
	temp = [arr[0][0]]
        for i in range(0, len(arr) - 1):
		temp.append(arr[i+1][0] - arr[i][1])
        
        temp.append(total - arr[-1][1])

	return temp

def how_many_between(arr, center, side):
        return [max(((arr[i] - 2 * side) // center), 0) for i in range(0, len(arr))]

def make_start_list(bl):
	start_l = [0]
        for i in range(0, len(bl)):
		start_l.append(bl[i][1])

	return start_l

#generate data extraction list
def make_take_list(chrom, bl, seq_len, side_len, resolution, total_num, chrom_start):
        start_list = make_start_list(bl)
        
        bl_intervals = interval_calc(bl, total_num)
        
        center_bp = seq_len * resolution
        side_bp = side_len * resolution
        tot_bp = center_bp + 2 * side_bp
        
        allow_num = how_many_between(bl_intervals, center_bp, side_bp)
        
        #calculate the location of each region
        location_list = []
        
        total_data_count = chrom_start
        
        for i in range(0, len(start_list)):
                start_head = start_list[i]
                tot_allow = allow_num[i]
                
                for j in range(0, tot_allow):
                        head = start_head + j * center_bp
                        tail = head + tot_bp
                        
                        location_list.append([chrom, head, tail, total_data_count])
                        
                        total_data_count += seq_len + side_len * 2

	return location_list, total_data_count

def peak_reading(fn):
        peaks = []
        f = open(fn, 'r')
        
        for line in f:
                sl = line.split()
                peaks.append(float(sl[1]))
        
        f.close()
        return peaks

def arr_to_str(arr):
	temp = []
        for i in range(0, len(arr)):
		temp.append(str(arr[i]))

	return ' '.join(temp)

#write y data
def write_y(y, fname):
        f = open(fname, 'w')
        for i in range(0, len(y)):
		f.write(str(i) + " " + str(y[i]) + " \n")

	f.close()

def save_list(read_list, fname):
        f = open(fname, 'w')
        for i in range(0, len(read_list)):
                f.write(arr_to_str(read_list[i]) + " \n")
        
        f.close()

def load_list(fname):
	temp = []
	f = open(fname, 'r')

        for line in f:
		sl = line.split()
                t = [sl[0]]
                for i in range(1, len(sl)):
			t.append(int(sl[i]))

		temp.append(t)

	return temp

def get_middle(mat, side_len, whole_len, ax):
	return np.split(mat, [side_len, whole_len - side_len], axis=ax)[1]

def extract_list(whole_list, sub):
	sub_list = []
        for i in range(0, len(sub)):
		sub_list.append(whole_list[sub[i]])

	return sub_list







