class TransformFeatures:

    def __init__(self, data_in, na_var_list, numerical_var_list=numerical,
                 cutoff_missing=0.25, cutoff_fill=0.05, fill_method='mode',
                 print_details=True):
        """ Constructor method
        """
        self.data_in = data_in
        self.na_var_list = na_var_list
        self.numerical_var_list = numerical_var_list

        data_out = pd.DataFrame()
        dropped_cols = []
        self._drop_missing()
        data_out=self._impute_values()
        data_out=self._convert_categories()

        
        return data_out

    def _drop_missing(self):
        missing_info = self.data_in.isna().sum()/len(self.data_in)
        for col in self.numerical_var_list:
            p_miss = missing_info[missing_info.index == col][0]
            if p_miss > cutoff_missing:
                if print_details == True:
                    print(col + ' - dropped because of missing values exceeding ' + str(cutoff_missing) + '%.' +
                        ' Missing values = '
                        + str(round(p_miss*100,2)) + '%.')
                dropped_cols.append(col)

    def _impute_values(self):
        for col_n in data_in.select_dtypes('number'):
        p_miss = missing_info[missing_info.index == col_n][0]
        if ((p_miss <= cutoff_fill) & (p_miss > 0)):
            if fill_method == 'mode':
                fill=data_in[col_n].mode()[0]
            elif fill_method == 'mean':
                fill=np.mean(data_in[col_n])
            elif fill_method == 'median':
                fill=np.median(data_in[col_n])
            else:
                if print_details == True:
                    print(fill_method + ' is not known. Column will not be transformed')
                continue
            data_out[col_n+'_NA'] = np.where(data_in[col_n].isnull(), 1, 0) # Adding variable indicating missing value
            data_out[col_n] = data_in[col_n].fillna(value = fill)
            if print_details == True:
                print(col_n + ' - ' + str(round(p_miss*100,4)) + '% of missing values. They are replaced with '
                    + fill_method + ' value - ' + str(fill))
                print(col_n+'_NA' + ' is created to indicate missing values of original variable.')
            try:
                na_var_list.append(col_n+'_NA')
                if print_details == True:
                    print(col_n+'_NA added to na_var_list.')
            except:
                if print_details == True:
                    print(col_n+'_NA already in na_var_list.')
        else :
            data_out[col_n] = data_in[col_n]
            if print_details == True:
                print(col_n + ' - ' + str(round(p_miss*100,4)) + '% of missing values. Variable copied.')

    def _convert_categories(self):
        for col_c in data_in.select_dtypes(exclude='number'):
        data_out[col_c] = data_in[col_c].astype('category')


def transform_features(data_in, na_var_list, cutoff_missing = 0.25, cutoff_fill = 0.05, fill_method = 'mode', 
                      print_details = True):
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
    for col in numerical:
        p_miss = missing_info[missing_info.index == col][0]
        if p_miss > cutoff_missing:
            if print_details == True:
                print(col + ' - dropped because of missing values exceeding ' + str(cutoff_missing) + '%.' +
                      ' Missing values = '
                     + str(round(p_miss*100,2)) + '%.')
            dropped_cols.append(col)
    for col_n in data_in.select_dtypes('number'):
        p_miss = missing_info[missing_info.index == col_n][0]
        if ((p_miss <= cutoff_fill) & (p_miss > 0)):
            if fill_method == 'mode':
                fill=data_in[col_n].mode()[0]
            elif fill_method == 'mean':
                fill=np.mean(data_in[col_n])
            elif fill_method == 'median':
                fill=np.median(data_in[col_n])
            else:
                if print_details == True:
                    print(fill_method + ' is not known. Column will not be transformed')
                continue
            data_out[col_n+'_NA'] = np.where(data_in[col_n].isnull(), 1, 0) # Adding variable indicating missing value
            data_out[col_n] = data_in[col_n].fillna(value = fill)
            if print_details == True:
                print(col_n + ' - ' + str(round(p_miss*100,4)) + '% of missing values. They are replaced with '
                    + fill_method + ' value - ' + str(fill))
                print(col_n+'_NA' + ' is created to indicate missing values of original variable.')
            try:
                na_var_list.append(col_n+'_NA')
                if print_details == True:
                    print(col_n+'_NA added to na_var_list.')
            except:
                if print_details == True:
                    print(col_n+'_NA already in na_var_list.')
        else :
            data_out[col_n] = data_in[col_n]
            if print_details == True:
                print(col_n + ' - ' + str(round(p_miss*100,4)) + '% of missing values. Variable copied.')
    for col_c in data_in.select_dtypes(exclude='number'):
            data_out[col_c] = data_in[col_c].astype('category')
    return data_out
