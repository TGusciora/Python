import pandas as pd
import numpy as np

# https://www.sphinx-doc.org/en/master/usage/quickstart.html
# https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html
# https://medium.com/@richdayandnight/a-simple-tutorial-on-how-to-document-your-python-project-using-sphinx-and-rinohtype-177c22a15b5b


class TransformFeatures:
    """
    Transforming dataset to fix numerical variables issues with missing values.

    Create pandas dataframe with transformed numerical features to
    drop variables with high % of missing values, impute missing values
    for variables with low % of missing values nad convert non-numerical
    features to category type.

    Transformations:
    Dropping columns - if more than X% missing values (default: 25%)
    Imputing values - if max X% missing values (default: 25%) with
    provided method (default: mode). For imputed variables also binary
    variables (variable name + '_NA') created indicating if there was
    imputation for given record (value = 1). Imputation variables are
    added to self.na_var_list (list of imputation variables).

    Parameters
    ----------
    data_in : str
        DataFrame to analyze.
    numerical_var_list : []
        List of numeric variables to be analyzed and transformed.
    na_var_list : []
        Name of input discrete variable list. Binary var_name + "_NA" 
        variables are appended to that list.
    cutoff_missing : float64, default = 0.25
        Percentage of missing values used as cutoff point for dropping
        variable. If missings > cutoff_missing then drop variable from
        DataFrame.
    cutoff_fill : float64, default = 0.05
        Percentage of missing values used as cutoff point for filling
        missing variables with fill_method. If missings < cutoff_fill
        then replace missing values with fill_method and create new
        variable var_name + "_NA" which indicates rows with missing
        values for original variable.
    fill_method : str, default = 'mode'
        Filling method for missing values, when variable meets cutoff_fill
        criteria. Can choose from average, median, mode.
    print_details : bool, default = True
        Parameter controlling informative output. If set to false functiom
        will supress displaying of detailed information.

    Attributes
    ----------
    data_in : pandas DataFrame
        Dataset with features to be analyzed and transformed
    na_var_list : []
        Name of discrete variable list. Binary var_name + "_NA" variables
        are appended to that list.
    numerical_var_list : []
        list of numeric variables
    data_out : pandas DataFrame
        Dataset with transformed features
    dropped_cols : pandas DataFrame
        list of columns to be dropped because of too many missing values

    Notes
    -------------------
    Required libraries: \n
    * import pandas as pd \n
    * import numpy as np

    Methods
    -------
    output(self)
        Imports file_name and returns as pandas DataFrame.
    __init__(self, file_name)
        Constructor method.
    _drop_missing(self)
        Dropping variables with missing value % higher than cutoff_missing
        parameter value.
    _impute_values(self)
        If % of missing values are higher than cutoff_fill, then fills
        selected variables with give fill_method (available values : mode,
        mean, median). Also creates binary _NA variables where 1 indicates
        that there was value imputation made for particular record.
    _convert_categories(self)
        Convert non-numeric variables to "category" type.
    """

    def __init__(self, data_in, numerical_var_list, na_var_list,
                 cutoff_missing=0.25, cutoff_fill=0.05, fill_method='mode',
                 print_details=True):
        """ Constructor method
        """
        self.data_in = data_in
        self.na_var_list = na_var_list
        self.numerical_var_list = numerical_var_list
        self.data_out = pd.DataFrame()
        self.dropped_cols = []
        self.cutoff_missing = cutoff_missing
        self.cutoff_fill = cutoff_fill
        self.fill_method = fill_method
        self.print_details = print_details

    def output(self):
        """
        Generate transformed output.

        Function calculates missing value % for each variable in dataset. Then
        performs (in order): \n
        1) Establishing list of variables to drop with missing values
            exceeding cutoff_missing treshold
        2) Imputing values according to fill_method parameter for variables
            with missing values not exceeding treshold
        3) Converting non-numeric variables to "category" type
        4) Dropping columns calculated from point 1)

        Returns
        -------
        data_out : DataFrame
            DataFrame with transformed features.
        """
        self.missing_info = self.data_in.isna().sum()/len(self.data_in)
        self._drop_missing()
        self._impute_values()
        self._convert_categories()
        self.data_out.drop(self.dropped_cols)
        return self.data_out

    def _drop_missing(self):
        """
        Append variables with many missing values to dropped_cols list.

        Function checks for every variable if % of missing values exceeds
        cutoff_missing treshold. If it does, then adds variable name to
        dropped_cols list. Prints steps to terminal. This can be supressed
        with self.print_details = False.
        """
        for col in self.numerical_var_list:
            p_miss = self.missing_info[self.missing_info.index == col][0]
            if p_miss > self.cutoff_missing:
                if self.print_details is True:
                    print(col + ' - dropped because missing values exceeding '
                          + str(self.cutoff_missing) + '%.' +
                          ' Missing values = '
                          + str(round(p_miss*100, 2)) + '%.')
                self.dropped_cols.append(col)

    def _impute_values(self):
        """
        Impute calculated values for missing values in features.

        Function checks for every variable if % of missing values does not
        exceed cutoff_fill treshold. If it doesn't, for missing value records
        it imputes value from fill_method (mode, mean, median) and creates
        _NA variable indicating value imputation for this record (value = 1).
        Prints steps to terminal. This can be supressed with self.print_details
        = False.
        """
        for col_n in self.data_in.select_dtypes('number'):
            p_miss = self.missing_info[self.missing_info.index == col_n][0]
            if ((p_miss <= self.cutoff_fill) & (p_miss > 0)):
                if self.fill_method == 'mode':
                    fill = self.data_in[col_n].mode()[0]
                elif self.fill_method == 'mean':
                    fill = np.mean(self.data_in[col_n])
                elif self.fill_method == 'median':
                    fill = np.median(self.data_in[col_n])
                else:
                    if self.print_details is True:
                        print(self.fill_method
                              + ' is not known. Column won\'t be transformed')
                    continue
                # Adding variable indicating missing value
                self.data_out[col_n+'_NA'] = np.where(self.data_in[col_n].isnull(),
                                                      1, 0)
                self.data_out[col_n] = self.data_in[col_n].fillna(value=fill)
                if self.print_details is True:
                    print(col_n + ' - ' + str(round(p_miss*100, 4))
                          + '% of missing values. They are replaced with '
                          + self.fill_method + ' value - ' + str(fill))
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
        """
        Convert non-numeric variables to 'category' type.

        Selects all non-number type variables in self.data_out dataset
        and converts them to 'category' type.
        """
        for col_c in self.data_in.select_dtypes(exclude='number'):
            self.data_out[col_c] = self.data_in[col_c].astype('category')
