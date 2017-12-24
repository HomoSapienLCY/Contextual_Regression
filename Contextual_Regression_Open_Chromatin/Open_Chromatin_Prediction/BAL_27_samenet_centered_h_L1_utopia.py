from Reader import *

import tensorflow as tf
import tensorflow.python.ops.rnn_cell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.nn_ops import relu6

#list of data for training and testing
feature_mark_list = ["H2AK5ac", "H2BK120ac", "H2BK12ac", "H2BK15ac", "H2BK20ac", "H2BK5ac", "H3K14ac", "H3K18ac", "H3K23ac" ,"H3K4ac", "H3K56ac", "H4K5ac", "H4K8ac", "H4K91ac", "H3K27ac", "H2A.Z", "H3K4me1", "H4K20me1", "H3K79me1", "H3K23me2", "H3K4me2", "H3K79me2", "H3K4me3", "H3K27me3", "H3K9me3", "H3K36me3", "H3K9ac"]
target_mark = "DNase"

training_percentage = 0.7

mark_folder = "../data/"
mark_file_tail = ".pval.signal.bigwig"

num_feat = 27
num_layers = 2
#hidden_size=number of neurons
hidden_size = 40
#state_size is hidden_size * num_layers * 2
state_size = hidden_size * num_layers * 2
max_grad_norm = 5
learning_rate = 0.0001
batch_size = 10
num_of_epoch = 150
use_log_in_x = False

save_interval = 15

# read in instructions
if len(sys.argv) != 10:
        print("usage: python BAL_27_samenet_centered_other_cells.py cellline chromosome_names(separated by comma) resolution sequence_length side_length whether_to_use_available_sample_list keep_prob reg_scale utopia_scale")
        sys.exit(0)

cellline = sys.argv[1]
chrom_sp_str = sys.argv[2]
resolution = int(sys.argv[3])
seq_len = int(sys.argv[4])
side_len = int(sys.argv[5])
use_available_sample_list = str_to_bool(sys.argv[6])
keep_prob = float(sys.argv[7])
reg_scale = float(sys.argv[8])
utopia_scale = float(sys.argv[9])

#cut into name segments
chrom_sp_array = chrom_sp_str.split(',')

print("Work on " + chrom_sp_str)
print("Under resolution " + str(resolution))
print("Sequence length " + str(seq_len))
print("Side region length? " + str(side_len))
print("Use available list? " + str(use_available_sample_list))
print("Keep probability? " + str(keep_prob))
print("Regularization scale? " + str(reg_scale))

whole_len = seq_len + side_len * 2

center_block = seq_len * resolution
side_block = side_len * resolution
whole_block = center_block + side_block * 2

#make filename for reading
def fname_gen(markID):
        return mark_folder + cellline + "-" + markID + mark_file_tail

#read data from file
def read_data_from_list(mark_ID, read_list, nbins):
        data = []
        fn = fname_gen(mark_ID)
        fl = bw.open(fn)
        
        for i in range(0, len(read_list)):
                in_loc = read_list[i]
                
                chrom = in_loc[0]
                start = in_loc[1]
                end = in_loc[2]
                
                result = fl.stats(chrom, start, end, nBins=nbins, exact=True)
                
                filtered_result = [(ele if ele != None else 0) for ele in result]
                data.append(filtered_result)
        
        fl.close()
        
        return np.array(data)

#read data from file and also log the data
def read_data_from_list_log(mark_ID, read_list, nbins):
        data = []
        fn = fname_gen(mark_ID)
        fl = bw.open(fn)
        
        for i in range(0, len(read_list)):
                in_loc = read_list[i]
                
                chrom = in_loc[0]
                start = in_loc[1]
                end = in_loc[2]
                
                result = fl.stats(chrom, start, end, nBins=nbins, exact=True)
                
                filtered_result = [math.log((ele if ele != None else 0) + 1) for ele in result]
                data.append(filtered_result)
        
        fl.close()
        
        return np.array(data)

#load all the mark files
def load_all_marks(markID_list, read_list, nbins, log_or_not):
        if log_or_not:
        	read_function = read_data_from_list_log
        else:
        	read_function = read_data_from_list
        
        mark_pact = []
        for i in range(0, len(markID_list)):
		markID = markID_list[i]
		current_mark_peaks = read_function(markID, read_list, nbins)
		mark_pact.append(np.reshape(current_mark_peaks, [-1, nbins, 1]))

	return np.concatenate(mark_pact, axis=2)

#Neural Network
#---------------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, shape=[batch_size, whole_len, num_feat])
y_ = tf.placeholder(tf.float32, shape=[batch_size, seq_len])
is_training = tf.placeholder(tf.bool)

utopia_vec = tf.nn.l2_normalize(tf.constant(1.0, shape=[num_feat, 1], name="utopia_vec"), dim=0)

#generating initial weights
def weight_variable(shape, name):
        initial = tf.truncated_normal(shape=shape, stddev=0.1, name=name)
        return tf.Variable(initial)

def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(initial)

#drop action function
def drop_act_d():
        return tf.nn.dropout(x, keep_prob)

def drop_act_nd():
        return x / keep_prob

#cosine distance calculation, for utopia penalty
def calc_cos_dist(con_tensor):
        norm_con_tensor = tf.nn.l2_normalize(con_tensor, dim=1)
        sim_mat = tf.matmul(norm_con_tensor, utopia_vec)
        one = tf.constant(1.0)
        
        return one - tf.abs(sim_mat)

#dropout and reshape inputs
inp = tf.cond(is_training, drop_act_d, drop_act_nd)

inpu = [tf.squeeze(input_, [1]) for input_ in tf.split(axis=1, num_or_size_splits=whole_len, value=inp)]

def lstm_cell():
        return tf.nn.rnn_cell.LSTMCell(hidden_size)

def create_stacked_LSTM():
        return tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

# Create the internal multi-layer cell
forward_cell = create_stacked_LSTM()
backward_cell = create_stacked_LSTM()

fw_initial_state = forward_cell.zero_state(batch_size, tf.float32)
bw_initial_state = backward_cell.zero_state(batch_size, tf.float32)

#Context model
#outputs_bidirection has shape [batch_size, 2 * hidden_size] (10, 60), and length [whole_len] (50)
outputs_bidirection, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=forward_cell, cell_bw=backward_cell, inputs=inpu, initial_state_fw = fw_initial_state, initial_state_bw = bw_initial_state)

#Calculate context from bidirectional outputs
w_a_1 = weight_variable(name="w_a_1", shape=[hidden_size * 2, hidden_size])
b_a_1 = bias_variable(name="b_a_1", shape=[hidden_size])

#vertical net weight
w_a_v = weight_variable(name="w_a_v", shape=[hidden_size, seq_len])
b_a_v = bias_variable(name="b_a_v", shape=[seq_len])

#horizontal net weight
w_a_h = weight_variable(name="w_a_h", shape=[hidden_size, num_feat])
b_a_h = bias_variable(name="b_a_h", shape=[num_feat])

b = bias_variable(name="b", shape=[1])

#arrays to save v and h context
v_con_array = []
h_con_array = []
whole_con_array = []
cosine_penalty_array = []

#array to save y
y_array = []

for i in range(side_len, whole_len - side_len):
	current_output = outputs_bidirection[i]

	#multiply output by w_a and add b_a, get inner_sum of size [batch_size, hidden_size]
	inner_sum = tf.add(tf.matmul(current_output, w_a_1), b_a_1)
        con_i = tanh(inner_sum)

	#calculate the vertical (feature) and horizontal (distal) context vectors
	#shape [batch_size, seq_len]
	con_v = tf.nn.softmax(tf.add(tf.matmul(con_i, w_a_v), b_a_v))
        #shape [batch_size, num_feat]
        con_h = tf.add(tf.matmul(con_i, w_a_h), b_a_h)

	v_con_array.append(tf.expand_dims(con_v, 1))
        h_con_array.append(tf.expand_dims(con_h, 1))

	#tensor product each batch to generate the whole context con_vh
	tiled_con_v = tf.tile(tf.expand_dims(con_v, 2), tf.stack([1, 1, num_feat]))
        tiled_con_h = tf.tile(tf.expand_dims(con_h, 1), tf.stack([1, seq_len, 1]))
        #shape [batch_size, seq_len, num_feat]
        con_vh = tf.multiply(tiled_con_v, tiled_con_h)

	#calculate cosien_penalty, aka utopia penalty
	cosine_penalty = calc_cos_dist(con_h)
        cosine_penalty_array.append(cosine_penalty)
        
	whole_con_array.append(tf.expand_dims(con_vh, 1))

	#slice away the two sides of inp which are not included in the prediction model
	l_discard, center_inp, r_discard = tf.split(value=inp, num_or_size_splits=[i - side_len, seq_len, 3 * side_len - i], axis=1)
	#multiply (dot product)
	con_i = tf.multiply(center_inp, con_vh)

	#reduce sum to generate the prediction, con_i_o has shape [batch_size]
	con_i_o = tf.reduce_sum(con_i, axis=[1, 2]) + b
        y_array.append(tf.expand_dims(con_i_o, 1))

#calculate y_con and sum
y = tf.reshape(tf.concat(axis=1, values=y_array), [-1, 1])

#save the contexts for analysis
v_con = tf.reshape(tf.concat(axis=1, values=v_con_array), [-1, 1])
h_con = tf.reshape(tf.concat(axis=1, values=h_con_array), [-1, 1])
cp = tf.reshape(tf.concat(axis=1, values=cosine_penalty_array), [-1, 1])

#apply regularizations
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=reg_scale)
utopia_regularizer = tf.contrib.layers.l1_regularizer(scale=utopia_scale)

#calculate and add up the errors
error_cost = tf.sqrt(tf.reduce_mean(tf.square(y - tf.reshape(y_, [-1, 1]))))
regularization_cost = tf.contrib.layers.apply_regularization(l1_regularizer, [h_con])
utopia_cost = tf.contrib.layers.apply_regularization(utopia_regularizer, [cp])

cost = error_cost + regularization_cost + utopia_cost

#training
tvars = tf.trainable_variables()
gradi = tf.gradients(cost, tvars)
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.apply_gradients(zip(grads, tvars))

#---------------------------------------------------------------------------------------------------

#read blacklist, which are error prone data regions in ChIP-seq
blacklist_dict = {}

blacklist_file = open("wgEncodeDacMapabilityConsensusExcludable.bed", 'r')
for line in blacklist_file:
        sl = line.split()
        cname = sl[0]
        start = int(sl[1])
        end = int(sl[2])
        if cname in blacklist_dict.keys():
                blacklist_dict[cname].append([start, end])
        else:
                blacklist_dict[cname] = [[start, end]]

blacklist_file.close()

#generate the black list for removal
[blockized_blacklist_dict, blank_size_dict] = blacklist_blockizer(blacklist_dict, center_block)

read_list = []

chrom_start = 0

#create read dictionary
fl = bw.open(fname_gen(target_mark))
for i in range(0, len(chrom_sp_array)):
        chrom_sp_cu = chrom_sp_array[i]
        
        total_bin_num = int(fl.chroms(chrom_sp_cu))
        
        #remove blacklisted area
	bl = blockized_blacklist_dict[chrom_sp_cu]

	#generate take list from bl
	current_list, chrom_start = make_take_list(chrom_sp_cu, bl, seq_len, side_len, resolution, total_bin_num, chrom_start)

	read_list.extend(current_list)

fl.close()

save_list(read_list, "data_location_list")

#load x
data_x = load_all_marks(feature_mark_list, read_list, whole_len, use_log_in_x)
#load y
data_y = read_data_from_list(target_mark, read_list, whole_len)
data_y = get_middle(data_y, side_len, whole_len, 1)

print data_x.shape
print y.shape

if use_available_sample_list:
        train_list_cu = get_list_from_file("train_index_list")
        test_list_cu = get_list_from_file("test_index_list")
        discard_list_cu = get_list_from_file("discard_index_list")
else:
	y_sum = np.sum(data_y, axis=1)

	#calculate tpi
	tpi = top_percent_array_index(y_sum, 0.1)
	#from bpi, sample 30, 20, 10% from up middle down
	bpi_top_30 = list(set(top_percent_array_index(y_sum, 0.4)) - set(tpi))
	bpi_bottom_30 = list(set(range(0,len(y_sum))) - set(top_percent_array_index(y_sum, 0.7)))
	bpi_middle_30 = list(set(range(0,len(y_sum))) - set(tpi) - set(bpi_top_30) - set(bpi_bottom_30))
                
	bpi_train = list(set(sample_from_sets(bpi_top_30, 0.30)) | set(sample_from_sets(bpi_middle_30, 0.20)) | set(sample_from_sets(bpi_bottom_30, 0.10)))
                
	bpi_rest = list(set(range(0,len(y_sum))) - set(tpi) - set(bpi_train))
                
	tpi_train, tpi_test = sample_sets(tpi, training_percentage)
	bpi_discard, bpi_test = sample_sets(bpi_rest, training_percentage - 0.1) #0.6 when training

	#combine and shuffle
	train_list_cu = list(set(tpi_train) | set(bpi_train))
	test_list_cu = list(set(tpi_test) | set(bpi_test))
	discard_list_cu = bpi_discard

	random.shuffle(train_list_cu)
	random.shuffle(test_list_cu)
	random.shuffle(discard_list_cu)

	#cut off the tails
	train_list_cu = train_list_cu[:((len(train_list_cu) // batch_size) * batch_size)]
	test_list_cu = test_list_cu[:((len(test_list_cu) // batch_size) * batch_size)]
	discard_list_cu = discard_list_cu[:((len(discard_list_cu) // batch_size) * batch_size)]

	random.shuffle(train_list_cu)
	random.shuffle(test_list_cu)
	random.shuffle(discard_list_cu)
	
        #write selected datalist into file for later retrieval
	write_list_for_retrieval(train_list_cu, "train_index_list")
	write_list_for_retrieval(test_list_cu, "test_index_list")
	write_list_for_retrieval(discard_list_cu, "discard_index_list")

#pack data into training format
data_x_pack = [np.reshape(data_x[train_list_cu], [-1, batch_size, whole_len, num_feat]), np.reshape(data_x[test_list_cu], [-1, batch_size, whole_len, num_feat])]
data_y_pack = [np.reshape(data_y[train_list_cu], [-1, batch_size, seq_len]), np.reshape(data_y[test_list_cu], [-1, batch_size, seq_len])]

train_tot_size = len(data_y_pack[0])
test_tot_size = len(data_y_pack[1])

print("total training data size: ")
print(data_x_pack[0].shape)
print("total testing data size: ")
print(data_x_pack[1].shape)

#Save original data for comparison
for i in range(0, train_tot_size):
        write_peaks("original_peaks_train", data_y_pack[0][i].reshape(-1), i * batch_size * seq_len)

for i in range(0, test_tot_size):
        write_peaks("original_peaks_test", data_y_pack[1][i].reshape(-1), i * batch_size * seq_len)

saver = tf.train.Saver(max_to_keep = 1000)
steps_per_epoch = train_tot_size

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

	for epoch in range(num_of_epoch):
        	start = time.time()
                print("Epoch %d: " % epoch)

		for i in range(steps_per_epoch):
                        #every 100 steps, write out error for sanity check
        		if i % 100 == 0:
                		train_cost = cost.eval(feed_dict={x: data_x_pack[0][i], y_: data_y_pack[0][i], is_training: True})
                        	print("step %d, training accuracy %g"%(i, train_cost))
                
                	#randomly select a batch and train
                        ri = random.randint(0, train_tot_size - 1)
                        x_current_train = data_x_pack[0][ri]
                        y_current_train = data_y_pack[0][ri]
                        
                        train_step.run(feed_dict={x: x_current_train, y_: y_current_train, is_training: True})
        
                end = time.time()
                print("this epoch takes a total of: " + str(end - start) + " seconds \n")

		#save data and context
		if epoch % save_interval == 0:
        		print("saving model at step " + str(epoch))
                
                	for i in range(0, train_tot_size):
                        	v_context, h_context, pred = sess.run(tuple([v_con, h_con, y]), feed_dict={x: data_x_pack[0][i], is_training: False})
                                suf = "train_" + str(epoch)
                                write_con_pred(v_context, h_context, pred, i * batch_size * seq_len * seq_len,  i * batch_size * seq_len * num_feat, i * batch_size * seq_len, suf)
                        
                        for i in range(0, test_tot_size):
                                v_context, h_context, pred = sess.run(tuple([v_con, h_con, y]), feed_dict={x: data_x_pack[1][i], is_training: False})
                                suf = "test_" + str(epoch)
                                write_con_pred(v_context, h_context, pred, i * batch_size * seq_len * seq_len,  i * batch_size * seq_len * num_feat, i * batch_size * seq_len, suf)
                
                        saver.save(sess, "Imputation_Model_" + str(epoch) + ".ckpt")
                        
                	print("model saved \n")

	#save data and context in the final step
	print("final model \n")

	for i in range(0, train_tot_size):
        	v_context, h_context, pred = sess.run(tuple([v_con, h_con, y]), feed_dict={x: data_x_pack[0][i], is_training: False})
                suf = "train_final"
                write_con_pred(v_context, h_context, pred, i * batch_size * seq_len * seq_len,  i * batch_size * seq_len * num_feat, i * batch_size * seq_len, suf)

        for i in range(0, test_tot_size):
                v_context, h_context, pred = sess.run(tuple([v_con, h_con, y]), feed_dict={x: data_x_pack[1][i], is_training: False})
                suf = "test_final"
                write_con_pred(v_context, h_context, pred, i * batch_size * seq_len * seq_len,  i * batch_size * seq_len * num_feat, i * batch_size * seq_len, suf)

	saver.save(sess, "Imputation_Model_final" + ".ckpt")
        
        print("model saved \n")

