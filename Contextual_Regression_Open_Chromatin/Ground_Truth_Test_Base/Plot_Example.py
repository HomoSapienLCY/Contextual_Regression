from Function import *

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

tot_data = 100000

ori_name = sys.argv[1]
pre_name = sys.argv[2]
num_example = int(sys.argv[3])
ID = int(sys.argv[4])

ori_arr = read_con(ori_name)
pre_arr = read_con(pre_name)

data_len = len(ori_arr[0])

index_arr = range(1, data_len+1)

o = ori_arr[26790]
p = pre_arr[26790]

plt.plot(index_arr, o)
plt.plot(index_arr, p)

o_sum = np.sum(ori_arr, axis=0)
p_sum = np.sum(pre_arr, axis=0)

print cos_dist(o_sum, p_sum)

plt.show()
