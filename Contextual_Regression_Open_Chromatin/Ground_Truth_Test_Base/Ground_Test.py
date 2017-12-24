from Function import *

import tensorflow as tf
import tensorflow.python.ops.rnn_cell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.nn_ops import relu6
from tensorflow.python.ops.nn_ops import relu

num_feat = 1
num_layers = 2
hidden_size = 80
max_grad_norm = 5
learning_rate = 0.0001
batch_size = 100
num_of_epoch = 201

save_interval = 5
sum_interval = 2

keep_prob = 1

#Neural Network Portion
#------------------------------------------------------------
x = tf.placeholder(tf.float32, shape=[batch_size, seq_len, num_feat])
y_ = tf.placeholder(tf.float32, shape=[batch_size])
is_training = tf.placeholder(tf.bool)

#drop action function
def drop_act_d():
        return tf.nn.dropout(x, keep_prob)

def drop_act_nd():
        return x / keep_prob

#dropout and reshape inputs
inp = tf.cond(is_training, drop_act_d, drop_act_nd)

inpu = [tf.squeeze(input_, [1]) for input_ in tf.split(axis=1, num_or_size_splits=seq_len, value=inp)]

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
#outputs_bidirection has shape [batch_size, 2 * hidden_size] (20, 400), and length [seq_len] (100)
outputs_bidirection, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=forward_cell, cell_bw=backward_cell, inputs=inpu, initial_state_fw = fw_initial_state, initial_state_bw = bw_initial_state)

#Calculate context from bidirectional outputs
w_a_1 = tf.get_variable("w_a_1", [hidden_size * 2, 1])
b_a_1 = tf.get_variable("b_a_1", [1])

a_con_array = []
y_con_array = []

for i in range(0, len(outputs_bidirection)):
        current_output = outputs_bidirection[i]

	#shape [batch_size, 1]
	con = tf.add(tf.matmul(current_output, w_a_1), b_a_1)
        
	a_con_array.append(con)

#shape [batch_size, seq_len]
a_con = tf.expand_dims(tf.concat(axis=1, values=a_con_array), 2)

#shape [batch_size, seq_len]
y_con = tf.multiply(inp, a_con)
#shape [batch_size]
y_con_sum = tf.squeeze(tf.reduce_sum(y_con, axis=1))

error_cost = tf.sqrt(tf.reduce_mean(tf.square(y_con_sum - y_)))
cost = error_cost

tvars = tf.trainable_variables()
gradi = tf.gradients(cost, tvars)
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.apply_gradients(zip(grads, tvars))
#------------------------------------------------------------

data_to_fit = read_data("generated_data")
batched_data = into_batchs(data_to_fit, batch_size)

tot_len = len(batched_data)
train_len = int(tot_len * training_percentage)
test_len = tot_len - train_len

tot_list = range(0, tot_len)

train_list = tot_list[0:train_len]
test_list = list(set(tot_list) - set(train_list))

steps_per_epoch = train_len / 10

write_with_list(batched_data, train_list, "original_train")
write_with_list(batched_data, test_list, "original_test")

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

	for epoch in range(num_of_epoch):
        	start = time.time()
                print("Epoch %d: " % epoch)
                
                for i in range(steps_per_epoch):
                        if i % 5 == 0:
                                index = train_list[i]
                                train_cost = cost.eval(feed_dict={x:  np.expand_dims(batched_data[index][0], axis=2), y_: batched_data[index][1], is_training: True})
                                print("step %d, training accuracy %g"%(i, train_cost))
                
                        ri = random.randint(0, train_len - 1)
                        index = train_list[ri]
                        x_current_train = batched_data[index][0]
                        y_current_train = batched_data[index][1]
        
                	train_step.run(feed_dict={x: np.expand_dims(x_current_train, axis=2), y_: y_current_train, is_training: True})
                
                end = time.time()
        	print("this epoch takes a total of: " + str(end - start) + " seconds \n")

		if epoch % save_interval == 0:
        		print("saving model at step " + str(epoch))
                
                	train_con_result = []
                        test_con_result = []
                        train_y_result = []
                        test_y_result = []
                
                	for i in range(0, train_len):
                                index = train_list[i]
                        	weight, pred = sess.run(tuple([a_con, y_con_sum]), feed_dict={x: np.expand_dims(batched_data[index][0], axis=2), is_training: False})
                        	train_con_result.append(weight)
                        	train_y_result.append(pred)
                        
                        for i in range(0, test_len):
                                index = test_list[i]
                                weight, pred = sess.run(tuple([a_con, y_con_sum]), feed_dict={x: np.expand_dims(batched_data[index][0], axis=2), is_training: False})
                                test_con_result.append(weight)
                                test_y_result.append(pred)
                                        
                        train_con_result = np.concatenate(train_con_result, axis = 0)
			test_con_result = np.concatenate(test_con_result, axis = 0)
        		train_y_result = np.expand_dims(np.concatenate(train_y_result, axis = 0), axis=1)
                	test_y_result = np.expand_dims(np.concatenate(test_y_result, axis = 0), axis=1)
                                
                        write_np_arr(np.squeeze(train_con_result), "train_" + str(epoch) + "_con")
                        write_np_arr(np.squeeze(test_con_result), "test_" + str(epoch) + "_con")
                        write_np_arr(train_y_result, "train_" + str(epoch) + "_y")
                        write_np_arr(test_y_result, "test_" + str(epoch) + "_y")
                        
                	print("model saved \n")

