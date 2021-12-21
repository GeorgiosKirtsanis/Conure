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


def main():
    # Define arguments
    print("Defining arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--datapath', type=str, default='Data/Session/original_desen_pretrain.csv',
                        help='data path')
    parser.add_argument('--datapath_index', type=str, default='Data/Session/index.csv',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=1,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=20000,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--has_positionalembedding', type=bool, default=False,
                        help='whether contains positional embedding before performing cnnn')
    parser.add_argument('--max_position', type=int, default=1000,
                         help='maximum number of for positional embedding, it has to be larger than the sequence lens')

    args = parser.parse_args()

    # Load Data using data_loader_t1.py
    print("Loading data...")
    dl = data_loader.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index})
    all_samples = dl.item
    items = dl.item_dict
    bigemb= dl.embed_len 

    # Randomly shuffle data
    print("Randomly shuffling data...")
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    if not os.path.isdir('data/T1'):
        os.makedirs('data/T1')

    # Run myopen
    f0 = open("data/T1/period_0.txt", "w")
    f1 = open("data/T1/period_1.txt", "w")
    f2 = open("data/T1/period_2.txt", "w")
    f3 = open("data/T1/period_3.txt", "w")
    f4 = open("data/T1/period_4.txt", "w")
    f5 = open("data/T1/period_5.txt", "w")
    f6 = open("data/T1/period_6.txt", "w")
    f7 = open("data/T1/period_7.txt", "w")
    f8 = open("data/T1/period_8.txt", "w")
    f9 = open("data/T1/period_9.txt", "w")
    f10 = open("data/T1/period_10.txt", "w")
    f11 = open("data/T1/period_11.txt", "w")
    f12 = open("data/T1/period_12.txt", "w")
    f13 = open("data/T1/period_13.txt", "w")
    f14 = open("data/T1/period_14.txt", "w")
    f15 = open("data/T1/period_15.txt", "w")
    f16 = open("data/T1/period_16.txt", "w")

    lines = all_samples.shape[0]
    length = all_samples.shape[1]
    limit = float(1 / 17)

    for i in range(0, lines):
        if (i>=int(0*limit*lines)) & (i<int(1*limit*lines)):
            f=f0
        if (i>=int(1*limit*lines)) & (i<int(2*limit*lines)):
            f=f1
        if (i>=int(2*limit*lines)) & (i<int(3*limit*lines)):
            f=f2
        if (i>=int(3*limit*lines)) & (i<int(4*limit*lines)):
            f=f3
        if (i>=int(4*limit*lines)) & (i<int(5*limit*lines)):
            f=f4
        if (i>=int(5*limit*lines)) & (i<int(6*limit*lines)):
            f=f5
        if (i>=int(6*limit*lines)) & (i<int(7*limit*lines)):
            f=f6
        if (i>=int(7*limit*lines)) & (i<int(8*limit*lines)):
            f=f7
        if (i>=int(8*limit*lines)) & (i<int(9*limit*lines)):
            f=f8
        if (i>=int(9*limit*lines)) & (i<int(10*limit*lines)):
            f=f9
        if (i>=int(10*limit*lines)) & (i<int(11*limit*lines)):
            f=f10
        if (i>=int(11*limit*lines)) & (i<int(12*limit*lines)):
            f=f11
        if (i>=int(12*limit*lines)) & (i<int(13*limit*lines)):
            f=f12
        if (i>=int(13*limit*lines)) & (i<int(14*limit*lines)):
            f=f13
        if (i>=int(14*limit*lines)) & (i<int(15*limit*lines)):
            f=f14
        if (i>=int(15*limit*lines)) & (i<int(16*limit*lines)):
            f=f15
        if (i>=int(16*limit*lines)) & (i<int(17*limit*lines)):
            f=f16
        for j in range(0, length):
            if (all_samples[i, j] != 1) :
                f.write(str(i+1) + ' ' + str(all_samples[i, j]) + '\n')
        
    # Run myclose
    f0.close()
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    f7.close()
    f8.close()
    f9.close()
    f10.close()
    f11.close()
    f12.close()
    f13.close()
    f14.close()
    f15.close()
    f16.close()


if __name__ == '__main__':
    main()
