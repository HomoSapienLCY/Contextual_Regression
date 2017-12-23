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

#neural network
#-----------------------------------------------------------------------------

def MNIST_model(features, labels, mode):
        #input features
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
        
        if mode == tf.estimator.ModeKeys.EVAL:
        	input_data = tf.constant(1.0) - input_layer
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

	logits = dense(inputs=dropout_out, units=10)

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

	#Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

	#train mode
	if mode == tf.estimator.ModeKeys.TRAIN:
        	optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
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
	mnist_classifier.train(input_fn=train_input_fn, steps=500, hooks=[logging_hook])

	#Evaluate
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn, hooks=[logging_hook])
	print(eval_results)

	#Predict
	pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, num_epochs=1, shuffle=False)
	predictions = mnist_classifier.predict(input_fn=eval_input_fn, hooks=[logging_hook])


if __name__ == "__main__":
        tf.app.run()















