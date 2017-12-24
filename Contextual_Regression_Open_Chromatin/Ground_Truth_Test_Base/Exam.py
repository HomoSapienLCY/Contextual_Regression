from Function import *

tot_data = 100000

epoch = sys.argv[1]

train_num = int(tot_data * training_percentage)
test_num = tot_data - train_num

y_pred_test = read_pred("test_" + epoch + "_y")
y_ori_test = read_pred("original_test")

y_pred_train = read_pred("train_" + epoch + "_y")
y_ori_train = read_pred("original_train")

print "train_error:"
print calc_diff(y_ori_train, y_pred_train)
print "test_error"
print calc_diff(y_ori_test, y_pred_test)

a_ori = read_con("generated_weights")
a_train_ori = a_ori[0:train_num]
a_test_ori = a_ori[train_num:]

a_train_pred = read_con("train_" + epoch + "_con")
a_test_pred = read_con("test_" + epoch + "_con")

errors = compare_con(a_train_ori, a_train_pred)

print "con_errors:"
print math.sqrt(np.dot(errors, errors) / len(errors))
