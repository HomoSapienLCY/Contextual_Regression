from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import random
import math
import time
import sys
import string
import shlex, subprocess

import numpy as np

import tensorflow as tf
from tensorflow.python.layers.convolutional import conv2d
from tensorflow.python.layers.pooling import max_pooling2d
from tensorflow.python.layers.core import dense
from tensorflow.python.layers.core import dropout
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.nn_ops import relu6
from tensorflow.python.ops.nn_ops import relu

tf.logging.set_verbosity(tf.logging.INFO)

hidden_size = 1024
drop_rate = float(sys.argv[1])
reg_scale = float(sys.argv[2])

#neural network
#-----------------------------------------------------------------------------

#generating initial weights
def weight_variable(shape, name):
        initial = tf.truncated_normal(shape=shape, stddev=0.2, name=name)
        return tf.Variable(initial)

def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(initial)

def MNIST_model(features, labels, mode):
        #input features
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
        
        if mode == tf.estimator.ModeKeys.EVAL:
        	#input_data = tf.constant(1.0) - input_layer
                input_data = input_layer
        else:
        	input_data = input_layer

	#convolution 1
	conv1 = conv2d(inputs=input_data, filters=32, kernel_size=[5, 5], padding="same", activation=relu, use_bias=True)

	#max pooling 1
        pool1 = max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	#convolution 2
	conv2 = conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=relu, use_bias=True)
        
        #max pooling 2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	#Fully connected
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
	dense_out = dense(inputs=pool2_flat, units=hidden_size, activation=relu)
        dropout_out = dropout(inputs=dense_out, rate=drop_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

	#Generate a [28 * 28, 10] matrix as context
	#initialize
	w_a_1 = weight_variable(name="w_a_1", shape=[hidden_size, 28 * 28 * 10])
        b_a_1 = bias_variable(name="b_a_1", shape=[28 * 28 * 10])
        
        context = tf.add(tf.matmul(dropout_out, w_a_1), b_a_1)
        context_matrix = tf.reshape(context, [-1, 28 * 28, 10])

	#dot product layer
	input_data_flat = tf.reshape(input_data, [-1, 28 * 28, 1])
        input_data_tiled = tf.tile(input=input_data_flat, multiples=[1, 1, 10])
        weighted_context = tf.multiply(input_data_tiled, context_matrix)

	#Generate softmax result
	logits = tf.reduce_sum(weighted_context, axis=[1])

	#a dictionary of prediction operators
        predictions = {
                "logits": tf.multiply(logits, tf.constant(1.0), name="logit_out"),
                #class prediction
                "classes": tf.argmax(input=logits, axis=1),
                #probability prediction
        	"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	#prediction mode
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	#regularization
	l1_regularizer = tf.contrib.layers.l1_regularizer(scale=reg_scale)
	regularization_cost = tf.contrib.layers.apply_regularization(l1_regularizer, [context_matrix])

	#Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        error_cost = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
        
        #total lost
        loss = regularization_cost + error_cost

	#train mode
	if mode == tf.estimator.ModeKeys.TRAIN:
        	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        	train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	#evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#-----------------------------------------------------------------------------

def main(unused_argv):
	#load data
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images # Returns np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images # Returns np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

	#create the estimator
	mnist_classifier = tf.estimator.Estimator(model_fn=MNIST_model, model_dir="./")
        #logging
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1000)

	#Train
	train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
	mnist_classifier.train(input_fn=train_input_fn, steps=1, hooks=[logging_hook])

	#Evaluate
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn, hooks=[logging_hook])
	print(eval_results)

	#Predict
	pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, num_epochs=1, shuffle=False)
	predictions = mnist_classifier.predict(input_fn=eval_input_fn, hooks=[logging_hook])


if __name__ == "__main__":
        tf.app.run()















