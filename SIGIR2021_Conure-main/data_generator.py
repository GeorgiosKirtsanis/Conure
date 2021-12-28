import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import data_loader_t1
import data_loader_neg
import data_loader_rand
import numpy as np
import argparse
import sys
import shutil


def main(dataset, datapath):
    # Define arguments
    print("Defining arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset, help='dataset')
    parser.add_argument('--datapath', type=str, default=datapath, help='data path')
    parser.add_argument('--datapath_index', type=str, default='Data/Session/index.csv', help='data path')
    parser.add_argument('--rho', type=float, default=0.3, help='static sampling in LambdaFM paper')
    args = parser.parse_args()

    # Loading data
    print("Loading data...")
    if args.dataset == 'T1':
        dl = data_loader_t1.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index})
        all_samples = dl.item
    elif (args.dataset == 'T2') or (args.dataset == 'T3'):
        dl = data_loader_neg.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index, 'lambdafm_rho': args.rho})
        all_samples = dl.example
    elif (args.dataset == 'T4'):
        dl = data_loader_rand.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index})
        all_samples = dl.example

    # Randomly shuffle data
    print("Randomly shuffling data...")
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]
    print(all_samples[0])
    print(all_samples[10])
    print(all_samples[20])
    print(all_samples[30])

    # Create Directory
    path = os.path.join('data', dataset)
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

    # Open txt files
    f0 = open(os.path.join(path, "period_0.txt"), "w")
    f1 = open(os.path.join(path, "period_1.txt"), "w")
    f2 = open(os.path.join(path, "period_2.txt"), "w")
    f3 = open(os.path.join(path, "period_3.txt"), "w")
    f4 = open(os.path.join(path, "period_4.txt"), "w")

    # Setting limits
    lines = all_samples.shape[0]
    length = 100
    # Limit must devide the dataset into 4 equal sized sets with complete steps for train and validation no matter of batch size
    # limit / 5 / 4 % 1024 == 0
    # validation percentage = 0.2 = 1/5
    # periods to be generated = 4 (5th will be empty for full training of ADER)
    # 1024 to be the biggest batch size for tunning
    if args.dataset == 'T1':
        limit = 363520
    elif args.dataset == 'T2':
        limit = 348160
    elif args.dataset == 'T3':
        limit = 61440
    elif args.dataset == 'T4':
        limit = 363520

    for i in range(0, lines):
        # Setting f depending on the line
        if (i>=0) & (i<limit):
            f=f0
        elif (i>=limit) & (i<2*limit):
            f=f1
        elif (i>=2*limit) & (i<3*limit):
            f=f2
        else:
            f=f3
        # Passing all the items of the sequence
        for j in range(0, length):
            if (all_samples[i, j] != 1) :
                f.write(str(i+1) + ' ' + str(all_samples[i, j]) + '\n')
        # T2, T3 and T4 have an extra target at the end of the sequence
        if (dataset == 'T2') or (dataset == 'T3') or (dataset == 'T4'):
            f.write(str(i+1) + ' ' + str(all_samples[i, length+1]) + '\n')
        
    # Close txt files
    f0.close()
    f1.close()
    f2.close()
    f3.close()
    f4.close()


if __name__ == '__main__':
    main('T1', 'Data/Session/original_desen_pretrain.csv')
    main('T2', 'Data/Session/original_desen_finetune_click_nouserID.csv')
    main('T3', 'Data/Session/original_desen_finetune_like_nouserID.csv')
    main('T4', 'Data/Session/original_desen_gender.csv')
