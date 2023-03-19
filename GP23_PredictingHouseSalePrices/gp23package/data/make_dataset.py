import pandas as pd
import os


class MakeDataset:
    """ Class used to create dataset (pandas DataFrame) from imported
    raw file_name.

    Required libraries:
        * import pandas as pd
        * import os

    :parameter file_name: raw file name to be imported. Has to be in
        \\data\\raw folder.
    :type file_name: str
    :ivar file_name: file_name passed to the instance on creation
    :ivar data: created pandas DataFrame from imported raw source file
    :return: MakeDataset class file
    :rtype: object
    """

    def __init__(self, file_name):
        """ Constructor method
        """
        self.file_name = file_name
        self.data = self._import_dataset()

    def _import_dataset(self):
        """ Importing raw file from <project>\\data\\raw folder
        Establishing raw data relative location based script location

        :return: data - imported data file
        :rtype: pandas DataFrame
        """
        # Return main package / project directory
        absolute_path = os.path.abspath(os.path.join(__file__, "../../.."))
        # Subdirectory with raw data
        relative_path = "\\data\\raw\\"
        # Establish full path to raw file
        full_path = absolute_path + relative_path + self.file_name
        # Import raw file as data pandas DataFrame
        data = pd.read_csv(full_path, delimiter="\t")
        return data
