# Contextual_Regression
This project is related to paper: Contextual Regression: An Accurate and Conveniently Interpretable Nonlinear Model for Mining Discovery from Scientific Data (https://arxiv.org/abs/1710.10728), please contact chl556@ucsd.edu if you have any questions.

# How to Run the Simulated Data Code
We have provided the code to run simulated data in the folder Ground_Truth_Test_Base. In order to run it, first open the file Run_Ground_Truth_Test.bash. In which you will see the following line:
```bash
for i in `seq 1 ${nor}`; do current_path=/???/Ground_truth_noise_${noise_level}_test_${i}; mkdir ${current_path}; cp * ${current_path}; cd ${current_path}; python ${runpy} 100000 ${noise_level}; bash submit.bash; cd ${new_path}; done
```
The program set the path to /???/Ground_truth_noise_${noise_level}_test_${i}, so you should enter your own folder path into /???/. Then the program call "bash submit.bash" to submit the job into the cluster to have it running, so you need to change this command to whichever way you would like to run the program.

After making the above change, cd into the folder of Ground_Truth_Test_Base and run:
```bash
bash Run_Ground_Truth_Test.bash 10 0.7 1
```
Where "10" means 10 sets of simulated data will be generated and run, "0.7" means 70% noise included in the target value y and "1" means using a complex Ground Truth function which is the one we used in the paper.

After the computation is finished. You can use the script Exam_All.py to calculate the RMSD of y and the context weights:
```bash
python Exam_All.py 0.7 200 10
```
Where 0.7 is the noise level, 200 and 10 are respectively the total number of epoch and simulated sets you want to include in the analysis. The program will organize the statistics into a .csv file.

# How to Run the Histone Mark Prediction Code
First, download the data in .bigwig format using the bash scripts X_downloadData_X.sh. Then run the program with run_tf.bash: 
```bash
python BAL_27_samenet_centered_h_L1_utopia.py E003 chrX,chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22 200 100 50 False 1.0 0.0 0.0001 > result.log
```
E003 is the ID of the cell line, chrX-chr22 is the chromosomes used as the data set separated by comma. 200 is the resolution of data in basepair and 100 is the number of bins in each data segment, 50 is the number of neighbor bins. False tells the program to generate a new training and testing list while True reads in the training and testing list from the current folder. 1.0 is the dropout rate. 0.0 is the Lasso penalty parameter we put on the features. 0.0001 is the utopia penalty parameter.

This set up will perform computing on all chromosomes in a cell line and thus consumes a lot of computing resource. I would recommend run it on a strong computer or only use a few of the chromosomes.

# Evaluation the Performance of Prediction
You can use the script All_Online_calc.py to evaluate the prediction performance. This script will calculate the 4 evaluation metrics (Correlation, Match1, Catch1Obs and Catch1Imp) as mentioned in the paper by https://www.nature.com/articles/nbt.3157 . Simply run it with the following command:
```bash
python All_Online_calc.py folder_that_contains_the_data_files
```

# Tutorial of Converting Neural Network to Contextual Regression
Most neural network models with a vector output can be converted into contextual regression for feature analysis. Here, we use the convolutional neural network from tensorflow tutorial (https://www.tensorflow.org/tutorials/layers) to illustrate the process.

In the original tutorial, the output from the dropout wrapper is feed into a fully connected neural network (dense) to generate logit for prediction as shown below:
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
In each MNIST image, we have 28 * 28 features (pixels) and we have a total of 10 possible classes. Thus the context should be arranged into a [28 * 28, 10] matrix which represents the linear weight of each of the 28 * 28 features to the probability that it belongs to any of the 10 classes:
```python
context_matrix = tf.reshape(context, [-1, 28 * 28, 10])
```
Then, the matrix is dot-producted with the original picture to generate the weighted context, which is the contribution of each feature to the probability that it belongs to each of the 10 classes. This weighted context is what usually used for the analysis of feature contribution in our own projects. Then, the contribution of each pixel is summed to generate the output for prediction as what is done in the original data:
```python
#dot product layer
input_data_flat = tf.reshape(input_data, [-1, 28 * 28, 1])
input_data_tiled = tf.tile(input=input_data_flat, multiples=[1, 1, 10])
weighted_context = tf.multiply(input_data_tiled, context_matrix)

#Generate softmax result
logits = tf.reduce_sum(weighted_context, axis=[1])
```
Now, we have finished the programming of the output part. However, we would also like to put a Lasso penalty on the features selected so that the model we acquire is sparse and easily analyzable. Thus we apply an L1 regularization on the context matrix:
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
We compare this model with the original model from the tensorflow tutorial. Both models are first run with learning_rate=1e-3 for 500 epochs and then learning_rate=1e-4 for 1500 epochs (we did it manually, which you will need to do as well). Drop_rate is set to be 0.5 and any reg_scale under 0.00001 should be good. If your implementation is correct, you should get some similar numbers as shown below:

| Model             | Test Set Accuracy |
| ----------------- | ----------------- |
| CNN (TF tutorial) | 99.14%            |
| CNN (CR)          | 99.16%            |

# Tips of Using Contextual Regression
I listed some caveats for you to avoid, so that you won't get frustrated using my method and write a long complaining letter to you grandma and your grandma give me a 5 hours phone call about her concerns      (´Д｀)b    ヽ(´∀｀||)

In some cases, scaling of the data or the change of some hyperparameters is needed to resolve the vanishing or exploding gradient. Reducing the number of neurons or applying regularization on the weights in the network is recommended. This practice can reduce the flexibility of the embedding function and make the results more robust and succinct.

Standardizing the features is strongly recommended (This can be easily done with the StandardScaler() from the sklearn package). Contextual regression is robust against noise of random value as we have demonstrated in the paper, but not against noise of constant large value. In order to understand this suggestion, consider the following situation:

Suppose we have a set of data points and their corresponding target values and features that are unnormalized:

| Data Points | Target Value | Feature Vector |
| ----------- | ------------ | -------------- | 
| A           | 5            | (1, 11, 5, 8)  |
| B           | 8            | (4, 41, 2, -3) |
| C           | 7            | (3, 21, 3, -1) |
| D           | -5           | (9, 0, 2, -7)  |

We can train a neural network which represents a function f(x) that maps the feature vectors to their corresponding target value of the data point. Now suppose we add a noise feature that has value 999 in all the data points:

| Data Points | Target Value f(x) | Feature Vector (x)  |
| ----------- | ----------------- | ------------------- | 
| A           | 5                 | (1, 11, 5, 8, 999)  |
| B           | 8                 | (4, 41, 2, -3, 999) |
| C           | 7                 | (3, 21, 3, -1, 999) |
| D           | -5                | (9, 0, 2, -7, 999)  |

Then, if we apply the contextual regression with Lasso constraint, the embedding model will have a global minimum when outputing (0, 0, 0, 0, f(x)/999) as context weight, which is shown in the following table:

| Data Points | Target Value f(x) | Feature Vector (x)  | Context Output        |
| ----------- | ----------------- | ------------------- | --------------------- | 
| A           | 5                 | (1, 11, 5, 8, 999)  | (0, 0, 0, 0, 5/999)   |
| B           | 8                 | (4, 41, 2, -3, 999) | (0, 0, 0, 0, 8/999)   |
| C           | 7                 | (3, 21, 3, -1, 999) | (0, 0, 0, 0, 7/999)   |
| D           | -5                | (9, 0, 2, -7, 999)  | (0, 0, 0, 0, -5/999)  |

Thus, this situation has yielded a model that solely focus on noise and defeats the purpose of contextual regression. Standardizing the data will greatly reduce the effect of noises with constant value since they have very low variance across data points.
