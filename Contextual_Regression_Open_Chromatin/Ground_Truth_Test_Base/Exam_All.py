from Function import *

noise_level = sys.argv[1]

tot_data = 100000

epoch_start = 0
epoch_interval = 5
epoch_end = int(sys.argv[2]) + 1

sample_ID_start = 1
sample_ID_interval = 1
sample_ID_end = int(sys.argv[3]) + 1

folder_prefix = "../Ground_truth_noise_" + noise_level + "_test_"

train_num = int(tot_data * training_percentage)
test_num = tot_data - train_num

all_data = []
num_of_correct = sample_ID_end - sample_ID_start

for ID in range(sample_ID_start, sample_ID_end, sample_ID_interval):
        good_sample = True
	current_ID_data = []

	folder_name = folder_prefix + str(ID) + '/'

	o_test_fn = folder_name + "original_test"
        y_ori_test = read_pred(o_test_fn)
        
        o_train_fn = folder_name + "original_train"
        y_ori_train = read_pred(o_train_fn)
        
        o_weight_fn = folder_name + "generated_weights"
        a_ori = read_con(o_weight_fn)
        
        a_train_ori = a_ori[0:train_num]
	a_test_ori = a_ori[train_num:]
        
        for epoch in range(epoch_start, epoch_end, epoch_interval):
        	y_p_test_fn = folder_name + "test_" + str(epoch) + "_y"
		y_pred_test = read_pred(y_p_test_fn)

		y_p_train_fn = folder_name + "train_" + str(epoch) + "_y"
        	y_pred_train = read_pred(y_p_train_fn)

                a_p_test_fn = folder_name + "test_" + str(epoch) + "_con"
                a_test_pred = read_con(a_p_test_fn)
                
		a_p_train_fn = folder_name + "train_" + str(epoch) + "_con"
		a_train_pred = read_con(a_p_train_fn)

		p_error_train = calc_diff(y_ori_train, y_pred_train)
        	p_error_test = calc_diff(y_ori_test, y_pred_test)
                
                #if p_error_train > 0.99:
                	#num_of_correct -= 1
                	#print "1 error sample"
                	#good_sample = False
                	#break

		a_dev_train = compare_con(a_train_ori, a_train_pred)
        	a_dev_test = compare_con(a_test_ori, a_test_pred)

		a_error_train = math.sqrt(np.dot(a_dev_train, a_dev_train) / len(a_dev_train))
		a_error_test = math.sqrt(np.dot(a_dev_test, a_dev_test) / len(a_dev_test))

		current_ID_data.append([p_error_train, p_error_test, a_error_train, a_error_test])

        if good_sample:
		all_data.append(current_ID_data)

print "total ok data number: "
print num_of_correct

all_data = np.array(all_data)
print all_data.shape

all_data_mean = np.transpose(np.mean(all_data, axis = 0))
all_data_std = np.transpose(np.std(all_data, axis = 0))

print all_data_mean
print all_data_std

write_csv(all_data_mean, all_data_std, noise_level)


