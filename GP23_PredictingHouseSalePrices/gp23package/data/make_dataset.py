import pandas as pd
import os

def make_dataset(file_name = 'AmesHousing.tsv', dlm = "\t"):

    absolute_path = os.path.abspath(os.path.join(__file__ ,"../../.."))
    print(absolute_path)
    relative_path = "\\data\\raw"
    print(relative_path)
    data_dir = os.path.join(absolute_path, relative_path)
    print(data_dir)
    full_path = os.path.join(data_dir, file_name)
    print(full_path)
    #data = pd.read_csv(path = full_path, delimiter = dlm)
    #return data

data = make_dataset()
