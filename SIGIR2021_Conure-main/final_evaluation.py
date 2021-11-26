import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def main():
    metric_list = ['Accuracy', 'MRR5', 'HIT5', 'NDCG5']
    task_list = ['T1', 'T2']
    mode_list = ['pretrain', 'finetune']

    for metric in (metric_list):
        for task in (task_list):
            for mode in (mode_list):
                #Creating file path
                script_dir = os.path.dirname(__file__)

                #Running for 5 iterations
                my_metric = np.array([])
                for iteration in range(1, 6):
                    results_dir = os.path.join(script_dir, 'Data/Results/' + str(iteration) + '/' + task + '/' + mode + '/' + metric + '/')
                    if os.path.isdir(results_dir):
                        sample_file_name = task + '_' + mode + '_' + metric + '.csv'
                        full_file_name = results_dir + sample_file_name
                        df = pd.read_csv(full_file_name, header=None)
                        array = pd.DataFrame.to_numpy(df)
                        if iteration==1:
                            my_metric = array
                        else:
                            my_metric = np.append(my_metric, array, axis=0)
                    else:
                        break

                #Calculationg average and standard deviation
                std = np.std(my_metric, axis=0)
                avg = np.average(my_metric, axis = 0)
                steps = np.zeros_like(std)
                for i in range(len(steps)):
                    steps[i] = i + 1

                #Creating Figure
                fig = plt.figure(figsize = (10, 5))
                plt.plot(steps, avg, label = "Average")
                plt.plot(steps, std, label = "Standard Deviation")
                plt.xlabel("Steps")
                plt.ylabel(metric)
                title = task + " " + mode + ' | ' + 'Average and Standard Deviation of ' + metric + ' per Step'
                plt.title(title)
                plt.legend()

                #Saving Figure
                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, 'Data/Results/Final/' + task + '/' + mode + '/' + metric + '/')
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
                sample_file_name = task + '_' + mode + '_' + metric
                full_file_name = results_dir + sample_file_name
                plt.savefig(full_file_name)

                #Saving to .csv
                metric_values = np.array([avg.T, std.T])
                df = pd.DataFrame(metric_values)
                df.to_csv(full_file_name + '.csv', index=False, header=False)


if __name__ == '__main__':
    main()
