# Contextual_Regression
This project is related to paper: Contextual Regression: An Accurate and Conveniently Interpretable Nonlinear Model for Mining Discovery from Scientific Data (https://arxiv.org/abs/1710.10728)

# Tutorial of Converting Neural Network to Contextual Regression
Most neural network models with a vector output can be converted into contextual regression for feature analysis. Here, we use the convolutional neural network from tensorflow tutorial (https://www.tensorflow.org/tutorials/layers) to illustrate the process

In the original tutorial, the output from the dropout wrapper is input into a fully connected neural network (dense) to generate logit for prediction as shown below:
```python
#Fully connected
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense_out = dense(inputs=pool2_flat, units=hidden_size, activation=relu)
dropout_out = dropout(inputs=dense_out, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

#Generate the output for prediction
logits = dense(inputs=dropout_out, units=10)
```
To convert it into contextual regression, we remove this fully connected neural network and feed the output from dropout into a linear neural network that embeds the input into a linear model:
```python
#Embedding layer, generate a [28 * 28, 10] matrix as context
#initialize
w_a_1 = weight_variable(name="w_a_1", shape=[hidden_size, 28 * 28 * 10])
b_a_1 = bias_variable(name="b_a_1", shape=[28 * 28 * 10])
        
context = tf.add(tf.matmul(dropout_out, w_a_1), b_a_1)
```
In each MNIST image, we have 28 * 28 features (pixels) and we have a total of 10 possible classes. Thus the context should be arranged into a [28 * 28, 10] matrix which represent the linear weight of each one the of 28 * 28 features to the probability that it belongs to each of the 10 classes:
```python
context_matrix = tf.reshape(context, [-1, 28 * 28, 10])
```
Then, the matrix is dot-producted with the original picture to generate the weighted context, which is the contribution of each feature to the probability that it belongs to each of the 10 classes. This weighted context is what usually used for the analysis of feature contribution. Then, the contribution of each pixel is summed to generate the output for prediction as what is done in the original data:
```python
#dot product layer
input_data_flat = tf.reshape(input_data, [-1, 28 * 28, 1])
input_data_tiled = tf.tile(input=input_data_flat, multiples=[1, 1, 10])
weighted_context = tf.multiply(input_data_tiled, context_matrix)

#Generate softmax result
logits = tf.reduce_sum(weighted_context, axis=[1])
```
Now, we have finished the programming of the output part. However, we would also like to put regularization on the features selected so that the model we acquire is sparse and easily analyzable. Thus we apply an L1 regularization on the context matrix:
```python
#regularization
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=reg_scale)
regularization_cost = tf.contrib.layers.apply_regularization(l1_regularizer, [context_matrix])

#Calculate Loss (for both TRAIN and EVAL modes)
onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
error_cost = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

#total lost
loss = regularization_cost + error_cost
```
We compare this model with the original model from the tensorflow tutorial. Both models are first run with learning_rate=1e-3 for 500 epochs and then learning_rate=1e-4 for 1500 epochs. Droprate is set to be 0.5. At the end, they ended with very similar prediction performance. If your implementation is correct, you should get some similar numbers. We also have our version attached.

| Model             | Test Set Accuracy |
| ----------------- | ----------------- |
| CNN (TF tutorial) | 99.14%            |
| CNN (CR)          | 99.16%            |

# Caveat of Using Contextual Regression
In the applications in our own lab, we haven't experience too many problems. In some cases, scaling of the data or the change of some hyperparameters is needed to handle the vanishing or exploding gradient problem.
Standardizing the features is strongly recommended for the generation of sensable results (This can be easily done with the StandardScaler() from the sklearn package). In order to understand this suggestion, please consider the following situation:
Suppose we have a set of data points and their corresponding target values and features that are unnormalized:

| Data Points | Target Value | Feature Vector |
| ----------- | ------------ | -------------- | 
| A           | 5            | (1, 11, 5, 8)  |
| B           | 8            | (4, 41, 2, -3) |
| C           | 7            | (3, 21, 3, -1) |
| D           | -5           | (9, 0, 2, -7)  |

We can train a neural network which represents a function f(x) that maps the feature vectors to their corresponding target value of the data point. Now suppose we add a noise feature that has value 999 in all the data points:

| Data Points | Target Value f(x) | Feature Vector      |
| ----------- | ----------------- | ------------------- | 
| A           | 5                 | (1, 11, 5, 8, 999)  |
| B           | 8                 | (4, 41, 2, -3, 999) |
| C           | 7                 | (3, 21, 3, -1, 999) |
| D           | -5                | (9, 0, 2, -7, 999)  |

When we apply the contextual regression with Lasso constraint, the embedding model will have a global minimum when outputing (0, 0, 0, 0, f(x)/999) as context weight.

| Data Points | Target Value f(x) | Feature Vector      | Context Output        |
| ----------- | ----------------- | ------------------- | --------------------- | 
| A           | 5                 | (1, 11, 5, 8, 999)  | (0, 0, 0, 0, 5/999)   |
| B           | 8                 | (4, 41, 2, -3, 999) | (0, 0, 0, 0, 8/999)   |
| C           | 7                 | (3, 21, 3, -1, 999) | (0, 0, 0, 0, 7/999)   |
| D           | -5                | (9, 0, 2, -7, 999)  | (0, 0, 0, 0, -5/999)  |

Thus, this situation has yield an unusable model for interpretation and defeats the purpose of contextual regression.
