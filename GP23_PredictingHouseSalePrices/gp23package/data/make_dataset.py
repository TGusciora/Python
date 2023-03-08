import pandas as pd
import os


class MakeDataset:
    """ Create dataset (pandas DataFrame) from imported raw file_name

    :param file_name: string name of raw file to be imported. Has to be in
    \\data\\raw folder.

    :ivar file_name: file_name passed to the instance on creation
    :ivar data: created pandas DataFrame from imported raw source file
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.data = self._import_dataset()

    def _import_dataset(self):
        # Return main package / project directory
        absolute_path = os.path.abspath(os.path.join(__file__, "../../.."))
        # Subdirectory with raw data
        relative_path = "\\data\\raw\\"
        # Establish full path to raw file
        full_path = absolute_path + relative_path + self.file_name
        # Import raw file as data pandas DataFrame
        data = pd.read_csv(full_path, delimiter="\t")
        return data


# data = MakeDataset('AmesHousing.tsv').data
# print(data)
# print(type(data))
