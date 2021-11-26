import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

class Metric_Evaluation:
    def __init__(self, options):

        # Getting options values
        metric = options['metric']
        metric_values = options['metric_values']
        iters = options['iters']
        datapath = options['datapath']
        task = options['task']
        mode = options['mode']
        
        #Creating Figure
        fig = plt.figure(figsize = (10, 5))
        plt.plot(iters, metric_values)
        plt.xlabel("Steps")
        plt.ylabel(metric)
        title = task + " " + mode + ' | ' + metric + " per step for test set of " + datapath
        plt.title(title)

        #Creating Dictionaries and files
        for iteration in range(1, 6):
            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, 'Data/Results/' + str(iteration) + '/' + task + '/' + mode + '/' + metric + '/')
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

                sample_file_name = task + '_' + mode + '_' + metric
                full_file_name = results_dir + sample_file_name

                #Saving Figure
                plt.savefig(full_file_name)
                #Saving Array to csv
                df = pd.DataFrame(metric_values).T
                df.to_csv(full_file_name + '.csv', index=False, header=False)
                break