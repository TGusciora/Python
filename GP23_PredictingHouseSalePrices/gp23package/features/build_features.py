import pandas as pd
import numpy as np


class FeatureCorrection:
    """
    Transforming dataset to fix numerical variables issues with missing values.

    Create pandas dataframe with transformed numerical features to
    drop variables with high % of missing values, impute missing values
    for variables with low % of missing values nad convert non-numerical
    features to category type.

    Transformations:
    Dropping columns - if more than X% missing values (default: 25%) \n
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


class FeatureBinning:
    """
    Binning categorical features for continuous target variable.

    Creates dictonary storing for each variable from var_list (has to be
    subset of X_df DataFrame) grouped to similar categories via continuous
    optimal binning function from optbinning library. This results in
    dictionary per variable containing optimal binning specification and
    Weight Of Evidence + Information Value calculations. WoE and IV values
    have been recalculated in this function, as currently in optbinning
    library WoE is calculated only for binary dependent variable. Formula for
    continuous calculation is taken from listendata.com (see: Source materials
    2.)

    Parameters
    ----------
    X_df : str
        DataFrame with independent variables to analyze.
    y : str
        Series with dependent variable. Must be continuous.
    Var_list : str
        List of variable names to create optimal bins for and calculate WoE/IV
        /Target encoders. Must be a subset of X_df
        columns.
    prebin_method : str, default = "quantile"
        Quoting source materials 2. : "The pre-binning method. Supported
        methods are “cart” for a CART decision tree,  “quantile” to generate
        prebins with approximately same frequency and “uniform” to generate
        prebins with equal width."
    cat_coff : float, default = 0.01
        If category size is less than cat_coff % (default 1%) of total
        population , then it will be grouped in separate group. All categories
        with size lesser than cat_coff % will be grouped together.
    n_bins : int, default = 10
        Max limit to number of bins (grouped categories).
    print_details : bool, default = True
        Parameter controlling informative output. If set to false function
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
    * from optbinning import ContinuousOptimalBinning

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

    References
    ----------
    Source materials: \n
    1. Binning library <http://gnpalencia.org/optbinning/binning_continuous.html> \n
    2. WOE <https://www.listendata.com/2019/08/WOE-IV-Continuous-Dependent.html>

    Returns
    -------
    optbin_dict : Dictionary
        optbin_dict[i]["optbin"] - metadata about transformation, parameter
        values
        optbin_dict[i]["bin_table"] - table containing information about
        grouped category variables with below variables:
                Bin - list of values representing grouped categories\n
                Count - number of observations in bin \n
                Count % - % of all observations in dataset \n
                Sum - sum of dependent variable by bin \n
                Mean - mean of dependent variable by bin \n
                Min - minimum value of dependent variable by bin \n
                Max - maximum value of dependent variable by bin \n
                Zeros count - "0" values of dependent variable by bin \n
                WoE - Weight of Evidence (See: source materials 2).\n
                IV - Information Value specyfiying prediction power by bin\n
    """
    def __init__(self, X_df, y, var_list, prebin_method="quantile",
                 cat_coff=0.01, n_bins=10, metric="WoE", print_details=True):
        """ Constructor method
        """
        self.optibn_dict = {}
        self.target_sum = y.sum()
        self.var_list = var_list
        self.prebin_method = prebin_method
        self.X = X_df
        self.y = y
        self.cat_coff = cat_coff
        self.n_bins = n_bins
        self.metric = metric
        self.print_details = print_details
        self._optbin_create()
      
    def _optbin_create(self):
        for i in self.var_list:
            self.optbin_dict[i] = {}
            optbin = ContinuousOptimalBinning(name=i,
                                              prebinning_method=self.prebin_method,
                                              dtype="categorical",
                                              cat_cutoff=self.cat_coff,
                                              max_n_prebins=self.n_bins)
            self.optbin_dict[i]["optbin"] = self.optbin_dict[i]["optbin"].fit(self.X_df[i], self.y)
            bin_table = self.optbin_dict[i]["optbin"].binning_table.build()
            self.optbin_dict[i]["bin_table"] = bin_table
            self.optbin_dict[i]["bin_table"]["WoE"] = np.log((self.optbin_dict[i]["bin_table"]['Sum']/self.target_sum) /
                                                             self.optbin_dict[i]["bin_table"]['Count (%)'])
            self.optbin_dict[i]["bin_table"]["IV"] = ((self.optbin_dict[i]["bin_table"]['Sum']/self.target_sum) -
                                                      self.optbin_dict[i]["bin_table"]['Count (%)']) * self.optbin_dict[i]["bin_table"]["WoE"]
            self.optbin_dict[i]["bin_table"]["WoE"].fillna(0, inplace=True)
            self.optbin_dict[i]["bin_table"]["IV"].fillna(0, inplace=True)
            self.optbin_dict[i]["bin_table"].at['Totals', 'IV'] = self.optbin_dict[i]["bin_table"].iloc[:-1]['IV'].sum()
            IV_temp = self.optbin_dict[i]["bin_table"].at['Totals', 'IV']
            if self.print_details is True:
                if IV_temp <= 0.02:
                    print(i, '- IV suggests not useful for prediction - ', IV_temp)
                elif IV_temp <= 0.1:
                    print(i, '- IV suggests weak predictive power - ', IV_temp)
                elif IV_temp <= 0.3:
                    print(i, '- IV suggests medium predictive power - ', IV_temp)
                elif IV_temp <= 0.5:
                    print(i, '- IV suggests strong predictive power - ', IV_temp)
                else:
                    print(i, '- IV suggests suspicious predictive power -', IV_temp)
        return self.optbin_dict

    def transform(data, var_name, optbin_dict, metric="WoE",
                  print_details=True):
        """
        Transforming dataset variables using stored transformation dictionary.

        Transforming variable var_name from Dataframe data using optbin_dict
        to return encoded value of Weight of Evidence (WoE) or mean dependent
        value from corresponding bin (see Optbin_create function). Further
        possible enhancements: adding troubleshooting like try except
        instructions for values of metric outside ["WoE", "Mean"] list.
        Currently for every value that is not equal to "WoE" function will
        transform variable using "Mean"
        metric.

        Parameters
        ----------
        data : str
            DataFrame with variables to transform.
        var_name : str
            Variable name (dtype = categorical) from data that will be
            transformed for corresponding metric from optbin_dict.
        optbin_dict : str
            Dictionary name from Optbin_create function. Storing information
            about optimal bins (grouped categories) and corresponding Weight
            of Evidence (WoE) and mean dependent variable value from bin.
        metric : str, default = 'WoE'
            Metric to be returned instead of var_name category. Returns
            optbin_dict[var_name]["bin_table"]["WoE"] for "Woe" value or
            optbin_dict[var_name]["bin_table"]["Mean"] for "Mean" value.
        print_details : bool, default = True
            Parameter controlling informative output. If set to false
            function will supress displaying of detailed information.
            
        Returns
        -------
        transformed : float
            Series representing transformed variable from category to float 
            values.
        """
        optbin = optbin_dict[var_name]["bin_table"]
        clean = optbin[['Bin','Mean','WoE','Count']].explode('Bin')
        clean.reset_index(drop = True, inplace = True)
        clean = clean[(clean.Count != 0) & (clean.Bin != "")]
        clean.drop(columns = 'Count', inplace = True)
        clean.loc['Other'] = clean.iloc[-1]
        clean.at['Other', 'Bin']='Other'
        if metric == 'WoE':
            dict_clean=clean.set_index("Bin")["WoE"].to_dict()
        else:
            dict_clean=clean.set_index("Bin")["Mean"].to_dict()           
        if print_details == True:                    
            for i in pd.Series(data[var_name].unique()):
                if i not in dict_clean:
                    print(var_name,':',i,' - value is not present in the binning table')
        transformed = data[var_name].map(dict_clean).fillna(dict_clean['Other'])
        return transformed
