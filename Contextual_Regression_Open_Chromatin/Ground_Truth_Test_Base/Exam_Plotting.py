import os.path

import sys
import string
import shlex, subprocess
import math
from math import pow
from math import log
import json
import operator
import random
import time

from Function import *

import numpy as np
import pyBigWig as bw
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

noise_level = sys.argv[1]

def read_csv(fn):
	f = open(fn, 'r')
        data = []
        for line in f:
                sl = line.strip()[:-1].split(',')
                if len(sl) > 1:
			data.append(str_to_float(sl))

	return (data[:4], data[4:])

means, stds = read_csv("../mean_and_std_" + noise_level + ".csv")

epoch_ind = range(0, 201, 5)

#plt.plot(epoch_ind, means[1], 'b', epoch_ind, means[3], 'r', alpha=0.5)
plt.errorbar(epoch_ind[:], means[1][:], stds[1][:], color = 'r', alpha=0.5)
#add expected error
plt.errorbar(epoch_ind[:], means[3][:], stds[3][:], color = 'b', alpha=0.5)
plt.plot(epoch_ind[:], [float(noise_level) / 2] * 21, 'y--', alpha=0.5)
plt.title("Training Epoch vs Percent Error for Noise Level " + noise_level)
plt.xlabel("Training Epoch")
plt.ylabel("Percent Error")
plt.xticks(np.arange(0, 200, 10))
plt.yticks(np.arange(0, 1, 0.05))
axes = plt.gca()
axes.set_xlim([-2, 202])
axes.set_ylim([-0.1, 1.1])

test_error = mlines.Line2D([], [], color='r', ls='-', markersize=15, label='test_error')
weight_error = mlines.Line2D([], [], color='b', ls='-', markersize=15, label='weight_error')
expected_error = mlines.Line2D([], [], color='y', ls='--', markersize=15, label='expected_error ')
plt.legend(handles=[test_error, weight_error, expected_error])

plt.savefig("Error_Plot_Noise_Level_" + noise_level + ".png")

plt.show()


