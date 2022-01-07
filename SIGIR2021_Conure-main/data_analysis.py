import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

class Data_Analysis:
    def __init__(self, options):

        #Creating datapath and data
        data = options['data']
        datapath = options['datapath']
        task = options['task']
        path = options['path']

        #Creating data_analysis matrix
        data_analysis = np.zeros(10)

        #Computing the number of clicks per user
        for i in range (data.shape[0]):
            for j in range (data.shape[1]):
                if (data[i, j] != 1):
                    #Saving according to percentage
                    if (j > 0) & (j <= 10):
                        data_analysis[0] += 1
                    elif (j > 10) & (j <= 20):
                        data_analysis[1] += 1
                    elif (j > 20) & (j <= 30):
                        data_analysis[2] += 1
                    elif (j > 30) & (j <= 40):
                        data_analysis[3] += 1
                    elif (j > 40) & (j <= 50):
                        data_analysis[4] += 1
                    elif (j > 50) & (j <= 60):
                        data_analysis[5] += 1
                    elif (j > 60) & (j <= 70):
                        data_analysis[6] += 1
                    elif (j > 70) & (j <= 80):
                        data_analysis[7] += 1
                    elif (j > 80) & (j <= 90):
                        data_analysis[8] += 1
                    elif (j > 90) & (j <= 100):
                        data_analysis[9] += 1
                    break

        #Creating Figure
        clicks = list(['(0-10]', '(10-20]', '(20-30]', '(30-40]', '(40-50]', '(50-60]', '(60-70]', '(70-80]', '(80-90]', '(90-100]'])
        users = list(data_analysis)
        fig = plt.figure(figsize = (10, 5))
        plt.bar(clicks, users, color ='maroon', width = 0.4)
        plt.xlabel("Number of clicks per user")
        plt.ylabel("Number of users")
        title = task + " | Data Analysis Plot for " + datapath
        plt.title(title)

        #Creating Dictionaries and files
        results_dir = os.path.join(path, task)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        full_file_name = os.path.join(results_dir, 'data_analysis')
                
        #Saving Figure and csv
        plt.savefig(full_file_name)
        df = pd.DataFrame(data_analysis).T
        df.to_csv(full_file_name + '.csv', index=False, header=False)