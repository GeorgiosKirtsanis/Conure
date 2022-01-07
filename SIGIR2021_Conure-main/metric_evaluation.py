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
        path = options['path']
        
        #Creating Figure
        fig = plt.figure(figsize = (10, 5))
        plt.plot(iters, metric_values)
        plt.xlabel("Steps")
        plt.ylabel(metric)
        title = task + " " + mode + ' | ' + metric + " per step for test set of " + datapath
        plt.title(title)

        results_dir = os.path.join(path, task, mode, 'metrics')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        full_file_name = os.path.join(results_dir, metric)

        #Saving Figure
        plt.savefig(full_file_name)
        #Saving Array to csv
        df = pd.DataFrame(metric_values).T
        df.to_csv(full_file_name + '.csv', index=False, header=False)
    