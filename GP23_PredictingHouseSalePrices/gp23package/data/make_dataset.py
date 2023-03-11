import pandas as pd
import os

# https://thomas-cokelaer.info/tutorials/sphinx/docstring_python.html - 
# for how to docstring documents
# Let's have this project in Sphinx - it will be easier than writing 

# mkdocs - https://towardsdatascience.com/five-tips-for-automatic-python-documentation-7513825b760e
# For me  - mkdocks, because I can host my webpage on github pages 
# - don't need to rely on external hosting website

class MakeDataset:
    """ Create dataset (pandas DataFrame) from imported raw file_name

    :param file_name: raw file name to be imported. Has to be in
    \\data\\raw folder.
    :type file_name: str

    :ivar file_name: file_name passed to the instance on creation
    :ivar data: created pandas DataFrame from imported raw source file
    """

    def __init__(self, file_name):
        """ Constructor method
        """
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
