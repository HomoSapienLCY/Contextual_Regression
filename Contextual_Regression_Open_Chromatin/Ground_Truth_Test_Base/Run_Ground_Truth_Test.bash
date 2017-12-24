#!/bin/bash

nor=$1
noise_level=$2
complexity=$3

if [ ${complexity} -eq 0 ]; then runpy=Testing_Peaks_Gen.py; else runpy=Testing_Peaks_Gen_Complex_Exp.py; fi

ori_path=/home/cliu556/tensorflow_run/Ground_Truth_Test_Base
new_path=${ori_path}_${noise_level}

cd ..; mv ${ori_path} ${new_path}; cd ${new_path};


for i in `seq 1 ${nor}`; do current_path=/home/cliu556/tensorflow_run/Ground_truth_noise_${noise_level}_test_${i}; mkdir ${current_path}; cp * ${current_path}; cd ${current_path}; python ${runpy} 100000 ${noise_level}; bash submit.bash; cd ${new_path}; done


