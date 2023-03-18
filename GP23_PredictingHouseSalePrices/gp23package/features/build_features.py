"""
.. module:: build_features
        :platform: Unix, Windows
        :synopsis: Module containing classes and functions for data 
        transformation.

.. moduleauthor:: Tomasz G <invalid@invalid.com>
"""

import pandas as pd
import numpy as np

# https://www.sphinx-doc.org/en/master/usage/quickstart.html
# https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html
# https://medium.com/@richdayandnight/a-simple-tutorial-on-how-to-document-your-python-project-using-sphinx-and-rinohtype-177c22a15b5b


class TransformFeatures:
    """ Create pandas dataframe with transformed numerical features to
    drop variables with high % of missing values, impute missing values
    for variables with low % of missing values nad convert non-numerical
    features to category type.

    Transofmations:
    Dropping columns - if more than X% missing values (default: 25%)
    Imputing values - if max X% missing values (default: 25%) with
    provided method (default: mode). For imputed variables also binary
    variables (variable name + '_NA') created indicating if there was
    imputation for given record (value = 1). Imputation variables are
    added to self.na_var_list (list of discrete variables).

    # Next - input mode and cutoffs as class variables (for inheritance 
    and ease of read)

    Required libraries:
    import pandas as pd
    import numpy as np

    :parameter file_name: raw file name to be imported. Has to be in
    \\data\\raw folder.
    :type file_name: str
    :ivar file_name: file_name passed to the instance on creation
    :ivar data: created pandas DataFrame from imported raw source file
    :return: imported data file
    :rtype: pandas DataFrame
    """

    def __init__(self, data_in, na_var_list, numerical_var_list):
        """ Constructor method
        """
        self.data_in = data_in
        self.na_var_list = na_var_list
        self.numerical_var_list = numerical_var_list
        self.data_out = pd.DataFrame()
        self.dropped_cols = []

    def output(self):
        self.missing_info = self.data_in.isna().sum()/len(self.data_in)
        self.print_details = True
        self._drop_missing()
        self._impute_values()
        self._convert_categories()
        self.data_out.drop(self.dropped_cols)
        return self.data_out

    def _drop_missing(self, cutoff_missing=0.25):
        for col in self.numerical_var_list:
            p_miss = self.missing_info[self.missing_info.index == col][0]
            if p_miss > cutoff_missing:
                if self.print_details is True:
                    print(col + ' - dropped because missing values exceeding '
                          + str(cutoff_missing) + '%.' +
                          ' Missing values = '
                          + str(round(p_miss*100, 2)) + '%.')
                self.dropped_cols.append(col)

    def _impute_values(self, cutoff_fill=0.05, fill_method='mode'):
        for col_n in self.data_in.select_dtypes('number'):
            p_miss = self.missing_info[self.missing_info.index == col_n][0]
            if ((p_miss <= cutoff_fill) & (p_miss > 0)):
                if fill_method == 'mode':
                    fill = self.data_in[col_n].mode()[0]
                elif fill_method == 'mean':
                    fill = np.mean(self.data_in[col_n])
                elif fill_method == 'median':
                    fill = np.median(self.data_in[col_n])
                else:
                    if self.print_details is True:
                        print(fill_method
                              + ' is not known. Column won\'t be transformed')
                    continue
                # Adding variable indicating missing value
                self.data_out[col_n+'_NA'] = np.where(self.data_in[col_n].isnull(),
                                                      1, 0)
                self.data_out[col_n] = self.data_in[col_n].fillna(value=fill)
                if self.print_details is True:
                    print(col_n + ' - ' + str(round(p_miss*100, 4))
                          + '% of missing values. They are replaced with '
                          + fill_method + ' value - ' + str(fill))
                    print(col_n+'_NA'
                          + ' is created to indicate missing values of'
                          + ' original variable.')
                try:
                    self.na_var_list.append(col_n+'_NA')
                    if self.print_details is True:
                        print(col_n+'_NA added to na_var_list.')
                except ValueError:
                    if self.print_details is True:
                        print(col_n+'_NA already in na_var_list.')
            else:
                self.data_out[col_n] = self.data_in[col_n]
                if self.print_details is True:
                    print(col_n + ' - ' + str(round(p_miss*100, 4))
                          + '% of missing values. Variable copied.')

    def _convert_categories(self):
        for col_c in self.data_in.select_dtypes(exclude='number'):
            self.data_out[col_c] = self.data_in[col_c].astype('category')


def transform_features(data_in, na_var_list, numerical_var_list,
                       cutoff_missing=0.25, cutoff_fill=0.05,
                       fill_method='mode', print_details=True):
    """
    Transform features based on their characteristics.

    Parameters
    ----------
    data_in : str
        DataFrame to analyze.
    cutoff_missing : float64, default = 0.25
        Percentage of missing values used as cutoff point for dropping variable. If missings > cutoff_missing then
        drop variable from DataFrame.
    cutoff_fill : float64, default = 0.05
        Percentage of missing values used as cutoff point for filling missing variables with fill_method. If 
        missings < cutoff_fill then replace missing values with fill_method and create new variable var_name + "_NA"
        which indicates rows with missing values for original variable
    fill_method : str, default = 'mode'
        Filling method for missing values, when variable meets cutoff_fill criteria. Can choose from average, median, mode.
    na_var_list : string
        Name of variable list for the binary var_name + "_NA" variables to be added to.
    seed : int
        Random number seed for results reproductibility.
    train_pct : float64, default = True 0.8
        Percentage of DS dataset to be used as train. 
    print_details : bool, default = True
        Parameter controlling informative output. If set to false function will supress displaying of detailed information.
    
    Returns
    -------
    data_out : DataFrame
        DataFrame with transformed features.
    """
    data_out = pd.DataFrame()
    missing_info = data_in.isna().sum()/len(data_in)
    dropped_cols = []
    for col in numerical_var_list:
        p_miss = missing_info[missing_info.index == col][0]
        if p_miss > cutoff_missing:
            if print_details is True:
                print(col + ' - dropped because of missing values exceeding '
                      + str(cutoff_missing) + '%.' +
                      ' Missing values = '
                      + str(round(p_miss*100,2)) + '%.')
            dropped_cols.append(col)
    for col_n in data_in.select_dtypes('number'):
        p_miss = missing_info[missing_info.index == col_n][0]
        if ((p_miss <= cutoff_fill) & (p_miss > 0)):
            if fill_method == 'mode':
                fill = data_in[col_n].mode()[0]
            elif fill_method == 'mean':
                fill = np.mean(data_in[col_n])
            elif fill_method == 'median':
                fill = np.median(data_in[col_n])
            else:
                if print_details is True:
                    print(fill_method + ' is not known. Column will'
                          +' not be transformed')
                continue
            # Adding variable indicating missing value
            data_out[col_n+'_NA'] = np.where(data_in[col_n].isnull(), 1, 0)
            data_out[col_n] = data_in[col_n].fillna(value=fill)
            if print_details is True:
                print(col_n + ' - ' + str(round(p_miss*100, 4)) 
                      + '% of missing values. They are replaced with '
                      + fill_method + ' value - ' + str(fill))
                print(col_n+'_NA' + ' is created to indicate '
                      + 'missing values of original variable.')
            try:
                na_var_list.append(col_n+'_NA')
                if print_details is True:
                    print(col_n+'_NA added to na_var_list.')
            except:
                if print_details is True:
                    print(col_n+'_NA already in na_var_list.')
        else:
            data_out[col_n] = data_in[col_n]
            if print_details is True:
                print(col_n + ' - ' + str(round(p_miss*100, 4))
                      + '% of missing values. Variable copied.')
    for col_c in data_in.select_dtypes(exclude='number'):
            data_out[col_c] = data_in[col_c].astype('category')
    return data_out
