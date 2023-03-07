import pandas as pd
import os


class MakeDataset:

    def __init__(self, file_name):
        self.file_name = file_name
        self.data = import_dataset(self.file_name)

    def import_dataset(file_name='AmesHousing.tsv', dlm="\t"):
        """ Create dataset (pandas DataFrame) for analysis

        :param file_name: raw source file to import
        :param dlm: delimiter used to separate columns in raw source file

        :return: pandas DataFrame representing imported raw source file
        """
        # Return main package / prooject directory
        absolute_path = os.path.abspath(os.path.join(__file__, "../../.."))
        # Subdirectory with raw data
        relative_path = "\\data\\raw\\"
        # Establish full path to raw file
        full_path = absolute_path + relative_path + file_name
        # Import raw file as data pandas DataFrame
        data = pd.read_csv(full_path, delimiter=dlm)
        return data


data = make_dataset()
