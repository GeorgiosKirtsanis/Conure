import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn
from collections import Counter


class Data_Loader:
    def __init__(self, options):

        #Creating dataset and index paths
        positive_data_file = options['dir_name']
        index_data_file = options['dir_name_index']

        #Create List of lines of dataset
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]

        #Transforming dataset from [0, X] to [1, Y] with dictrionary D = {'0': 1, ... , 'X': Y}
        max_document_length = max([len(x.split(",")) for x in positive_examples])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        self.item = np.array(list(vocab_processor.fit_transform(positive_examples)))
        self.item_dict = vocab_processor.vocabulary_._mapping

        #Saving dictionary size
        self.embed_len=len(self.item_dict)
        
        #Saving dictionary to index.csv
        f = open(index_data_file, 'w')
        f.write(str(self.item_dict))
        f.close()
        print("the index has been written to ", index_data_file)