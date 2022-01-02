import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import data_loader_t1 as data_loader
import generator_prune_t1 as generator_recsys
import data_analysis
import metric_evaluation
import math
import numpy as np
import argparse
import config
import sys


tf.set_random_seed(0)
size = [1 ,2, 3, 3]

weight = tf.get_variable('weight', initializer=tf.random_normal((size[0], size[1], size[2], size[3])))
# [[[[-1.4095545  -0.5366828  -0.5652379 ]
#    [ 0.526246   -0.11131065  0.26350743]
#    [ 0.80647576 -0.886015   -0.04653838]]
#   [[ 1.073006   -0.6044851  -0.7388869 ]
#    [ 1.2126138   0.06479809 -0.33498493]
#    [ 0.72790796 -0.778289    0.22196065]]]]

weight_norm = tf.abs(weight)
# [[[[1.4095545  0.5366828  0.5652379 ]
#    [0.526246   0.11131065 0.26350743]
#    [0.80647576 0.886015   0.04653838]]
#   [[1.073006   0.6044851  0.7388869 ]
#    [1.2126138  0.06479809 0.33498493]
#    [0.72790796 0.778289   0.22196065]]]]

weight_norm_onedim = tf.reshape(weight_norm, [size[0] * size[1] * size[2] * size[3]])
# [1.4095545  0.5366828  0.5652379  0.526246   0.11131065 0.26350743
#  0.80647576 0.886015   0.04653838 1.073006   0.6044851  0.7388869
#  1.2126138  0.06479809 0.33498493 0.72790796 0.778289   0.22196065]

cutoff_rank = 1 # For T1
top_k_weight_values = tf.nn.top_k(weight_norm_onedim, cutoff_rank + 1)
# [1.4095545 1.2126138]

top_k_weight = tf.nn.top_k(weight_norm_onedim, cutoff_rank + 1).values[cutoff_rank]
# 1.2126138

# Initialize zeros and ones and run tf.where
one = tf.ones_like(weight_norm)
zero = tf.zeros_like(weight_norm)
mask_weight = tf.where(weight_norm < top_k_weight, x=zero, y=one)
# [[[[1. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]
#   [[0. 0. 0.]
#    [1. 0. 0.]
#    [0. 0. 0.]]]]





########################## Case 1 ##########################

# 1) Extend cutoff for tf.nn.top_k by extention_1
extention_1 = 1
top_k_weight_values_1 = tf.nn.top_k(weight_norm_onedim, cutoff_rank + 1 + extention_1)
#[1.4095545 1.2126138 1.073006 ]
top_k_weight_1_important = tf.nn.top_k(weight_norm_onedim, cutoff_rank + 1 + extention_1).values[cutoff_rank + extention_1]
# 1.073006
mask_weight_1_important = tf.where(weight_norm < top_k_weight_1_important, x=zero, y=one)
# [[[[1. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]
#   [[1. 0. 0.]
#    [1. 0. 0.]
#    [0. 0. 0.]]]]

# 2) Add a propability_1 for deleting a percentage of neurons from the mask
random_mask_1 = tf.random_uniform(shape = weight_norm.shape)
# [[[[0.40284514 0.40805447 0.29488313]
#    [0.20003247 0.80905664 0.65500915]
#    [0.58685803 0.97063863 0.68234587]]
#   [[0.67275894 0.3114432  0.8062061 ]
#    [0.30782974 0.9795983  0.8974016 ]
#    [0.73439336 0.4037496  0.13495505]]]]
probability_1 = 0.4
mask_weight_1_important_probabled = tf.where(random_mask_1 < probability_1, x=zero, y=mask_weight_1_important)
# [[[[0. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]
#   [[0. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]]]
mask_weight_1_final = tf.identity(mask_weight_1_important_probabled)
# [[[[1. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]
#   [[1. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]]]







########################## Case 2 ##########################

# 1) Saving mask for same cutoff_rank
extention_2 = 1
top_k_weight_values_2 = tf.nn.top_k(weight_norm_onedim, cutoff_rank + 1 + extention_2)
#[1.4095545 1.2126138 1.073006 ]
top_k_weight_2_important = tf.nn.top_k(weight_norm_onedim, cutoff_rank + 1 + extention_2).values[cutoff_rank]
# 1.2126138
mask_weight_2_important = tf.where(weight_norm < top_k_weight_2_important, x=zero, y=one)
# [[[[1. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]
#   [[0. 0. 0.]
#    [1. 0. 0.]
#    [0. 0. 0.]]]]


# 2) Saving a percentage of neurons for next set of cutoff_rank:  [value(cutoff_rank) , value(cutoff_rank+extention_2)] 
top_k_weight_2_least_important = tf.nn.top_k(weight_norm_onedim, cutoff_rank + 1 + extention_2).values[cutoff_rank + extention_2]
# 1.073006
mask_weight_2_least_important = tf.where(((weight_norm < top_k_weight_2_important) & (weight_norm >= top_k_weight_2_least_important)), x=one, y=zero)
# [[[[0. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]
#   [[1. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]]]
random_mask_2 = tf.random_uniform(name = 'random_mask_2', shape = weight_norm.shape)
# [[[[0.08422506 0.41464472 0.56680906]
#    [0.82385874 0.46811152 0.7016159 ]
#    [0.8328512  0.13742697 0.7079787 ]]
#   [[0.31590486 0.38689983 0.67365825]
#    [0.7763928  0.4610814  0.05642426]
#    [0.60617614 0.2992779  0.7148559 ]]]]
probability_2 = 0.4
mask_weight_2_least_important_probabled = tf.where(random_mask_2 < probability_2, x=zero, y=mask_weight_2_least_important)
# [[[[0. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]
#   [[0. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]]]
mask_weight_2_final = tf.add(mask_weight_2_important, mask_weight_2_least_important_probabled)
# [[[[1. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]
#   [[0. 0. 0.]
#    [1. 0. 0.]
#    [0. 0. 0.]]]]




########################## Case 3 ##########################

# 1) Saving mask for cutoff_rank - 1
extention_3 = 1
top_k_weight_values_3 = tf.nn.top_k(weight_norm_onedim, cutoff_rank + 1 + extention_3)
#[1.4095545 1.2126138 1.073006 ]
top_k_weight_3_important = tf.nn.top_k(weight_norm_onedim, cutoff_rank).values[cutoff_rank - 1]
# 1.4095545
mask_weight_3_important = tf.where(weight_norm < top_k_weight_3_important, x=zero, y=one)
# [[[[1. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]
#   [[0. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]]]


# 2) Saving a percentage of neurons for next set of cutoff_rank:  [value(cutoff_rank) , value(cutoff_rank+extention_2)] 
top_k_weight_3_least_important = tf.nn.top_k(weight_norm_onedim, cutoff_rank + 1 + extention_2).values[cutoff_rank + extention_2]
# 1.073006
mask_weight_3_least_important = tf.where(((weight_norm < top_k_weight_3_important) & (weight_norm >= top_k_weight_3_least_important)), x=one, y=zero)
# [[[[0. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]
#   [[1. 0. 0.]
#    [1. 0. 0.]
#    [0. 0. 0.]]]]
random_mask_3 = tf.random_uniform(name = 'random_mask_3', shape = weight_norm.shape)
# [[[[0.4452772  0.76051927 0.06713736]
#    [0.90151644 0.37397754 0.6956866 ]
#    [0.6354872  0.37644994 0.7986016 ]]
#   [[0.3894441  0.6913122  0.57935   ]
#    [0.7406819  0.6351898  0.4150988 ]
#    [0.9792228  0.29356718 0.5677403 ]]]]
probability_3 = 0.4
mask_weight_3_least_important_probabled = tf.where(random_mask_3 < probability_3, x=zero, y=mask_weight_3_least_important)
# [[[[0. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]
#   [[0. 0. 0.]
#    [1. 0. 0.]
#    [0. 0. 0.]]]]
mask_weight_3_final = tf.add(mask_weight_3_important, mask_weight_3_least_important_probabled)
# [[[[1. 0. 0.]
#    [0. 0. 0.]
#    [0. 0. 0.]]
#   [[0. 0. 0.]
#    [1. 0. 0.]
#    [0. 0. 0.]]]]



with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())

    # # Initializer
    print('-------------------------')
    print('Weight Initializer')
    print(weight.eval())
    # print(weight_norm.eval())
    # print(weight_norm_onedim.eval())
    # print(top_k_weight_values.values.eval())
    # print(top_k_weight.eval())
    # print(mask_weight.eval())

    # Case 1
    print('-------------------------')
    print('Case 1 Mask')
    # print(top_k_weight_values_1.values.eval())
    # print(top_k_weight_1_important.eval())
    # print(mask_weight_1_important.eval())
    # print(random_mask_1.eval())
    # print(mask_weight_1_important_probabled.eval())
    print(mask_weight_1_final.eval())

    # # Case 2
    print('-------------------------')
    print('Case 2 Mask')
    # print(top_k_weight_values_2.values.eval())
    # print(top_k_weight_2_important.eval())
    # print(mask_weight_2_important.eval())
    # print(top_k_weight_2_least_important.eval())
    # print(mask_weight_2_least_important.eval())
    # print(random_mask_2.eval())
    # print(mask_weight_2_least_important_probabled.eval())
    print(mask_weight_2_final.eval())

    # # Case 3
    print('-------------------------')
    print('Case 3 Mask')
    # print(top_k_weight_values_3.values.eval())
    # print(top_k_weight_3_important.eval())
    # print(mask_weight_3_important.eval())
    # print(top_k_weight_3_least_important.eval())
    # print(mask_weight_3_least_important.eval())
    # print(random_mask_3.eval())
    # print(mask_weight_3_least_important_probabled.eval())
    print(mask_weight_3_final.eval())
