from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.compat import lzip
from IPython.display import display, HTML
import statsmodels as sm
import scipy.stats as stats
import statsmodels.stats.stattools as smt
import statsmodels.stats.diagnostic as smd
import matplotlib as plt
import seaborn as sn
import pandas as pd


class LinearDiagnostics():
    """
    Perform linear model diagnostics and statistical tests on given sample.

    For linear regression tasks. Given significance level as a parameter
    performs number of tests and plots several diagrams: \n
    * Descriptive statistics of model residuals \n
    * histogram of residuals \n
    * Kernel density plot of residuals \n
    * QQ plot of residuals \n
    * Jarque-Bera test for normality (residuals) \n
    * Shapiro-Wilk test for normality (residuals) \n
    * Anderson-Darling test for normality (residuals) \n
    * DAgostino and Pearsons normality test (residuals) - stats.normaltest \n
    * Durbin-Watson test for autocorrelation (residuals) \n
    * Predicted value vs residual scatter plot \n
    * Predicted value vs true value scatter plot \n
    * White test for homoscedasticity (residuals) \n
    * Breusch-Pagan test for homoscedasticity (residuals) \n
    * Variance Inflation Factor (VIF) table \n
    * Correlation matrix of independent variables

    Parameters
    ----------
    model_in : str
        Name of model stored in models_list dictionary
    data_in : str
        Dataset on which predictions will be made and residuals calculated.
        Must have variables required by the model present.
    target : str
        Series of target variable value. Must have corresponding index value
        with data_in.
    significance_level : float, default = 0.05
        Probability of rejecting H0 when it is true. Parameter for statistical
        tests. Against significance level we find critical value for
        statistical test and compare it to statistic.

    Notes
    -------------------
    Prerequisites: \n
    models_list - dictionary {'model_name' : { 'model' : class instance, \n
                                    'variables' : model variables set alias}}\n
    scoring_dict - dictionary { 'variables' : [corresponding list of variables]} \n
    Required libraries: \n
    * from statsmodels.stats.outliers_influence import variance_inflation_factor \n
    * from statsmodels.compat import lzip \n
    * from IPython.display import display, HTML \n
    * import statsmodels as sm \n
    * import scipy.stats as stats \n
    * import statsmodels.stats.stattools as smt \n
    * import statsmodels.stats.diagnostic as smd \n
    * import matplotlib as plt \n
    * import seaborn as sn \n
    * import pandas as pd
    """

    def __init__(self, model_in, data_in, target, significance_level=0.05):
        """
        Constructor method.
        """
        self.model = model_in
        self.data = data_in
        self.target = target
        self.alpha = significance_level
        # calculating residuals
        prdct_set = data_in[scoring_dict[model_list[self.model]['variables']]]
        self.predictions = model_list[self.model]['model'].predict(prdct_set)
        self.results_df = pd.DataFrame()
        self.results_df['predicted'] = list(self.predictions)
        self.results_df['actual'] = list(self.target)
        self.results_df['residual'] = self.results_df['predicted'] - self.results_df['actual']
        self.results_df = self.results_df.sort_values(by='residual').reset_index(drop=True)

    def residual_stats(self):
        """
        Calculate residual descriptive statistics on pointed dataset.
        """
        print('*** Test data residuals statistics ***')
        display(HTML(self.results_df.describe().to_html()))

    def res_kde(self):
        """
        Display residuals histogram.
        """
        sn.histplot(data=self.results_df['residual'])
        pylab.title('Residuals histogram')
        plt.show()

    def res_qq(self):
        """
        Display residuals qqplot.
        """
        stats.probplot(self.results_df['residual'], dist="norm", plot=pylab)
        pylab.title('Residuals qq plot')
        plt.show()

    def jb_normal_test(self):
        """
        Perform Jarque Bera test for normality of residuals
        """
        print('*** Jarque-Bera test for normality of residuals ***')
        name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
        jarqueBera_test = smt.jarque_bera(self.results_df['residual'])
        print(lzip(name, jarqueBera_test))
        if jarqueBera_test[1] > self.significance_level:
            print('On', self.significance_level, 'significance level we fail',
                  ' to reject H0 about normal distribution of residuals.')
        else:
            print('On', self.significance_level, 'significance level we ',
                  'reject H0 about normal distribution of residuals.')
        print()
    
    def sw_normal_test(self):
        """
        Perform Shapiro-Wilk test for normality of residuals.
        """
    # 'Shapiro-Wilk test for normality'
        print('Shapiro-Wilk test for normality')
        print(stats.shapiro(self.results_df['residual']))
        if stats.shapiro(self.results_df['residual'])[1] > self.alpha:
            print('On', self.alpha, 'significance level we fail to reject H0',
                  ' that residuals come from a normal distribution.')
        else:
            print('On', self.alpha, 'significance level we reject H0 that ',
                  'residuals come from a normal distribution.')
        print()

    def ad_normal_test(self):
        """
        
        """
        # Anderson-Darling Test for normality
        # H0 : sample is drawn from a population that follows a 
        # particular distribution.
        print('Anderson-Darling test for normality')
        res_and = stats.anderson(self.results_df['residual'], dist='norm')
        result = stats.anderson(self.results_df['residual'], dist='norm')
        print('Statistic: %.3f' % result.statistic)
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                print('Significance level - %.3f: %.3f (Critical Value) , ',
                      'fail to reject H0 that sample comes from normal ',
                      'distribution.' % (sl, cv))
            else:
                print('Significance level - %.3f: %.3f (Critical Value), ',
                      'rejecting H0 that sample comes from normal ',
                      'distribution.' % (sl, cv))
        print()

    def normal_test(self):
        """
        
        """
        #Normal test    
        print('Normal test for normality')
        stat, p = stats.normaltest(results_df['residual'])
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        if p > self.alpha:
            print('On', self.alpha,'significance level we fail to reject H0 that residuals come from a normal distribution.')
        else:
            print('On', self.alpha,'significance level we reject H0 that residuals come from a normal distribution.')
        print()

    def dw_autocorr_test(self):
        """
        
        """
        print('*** Durbin-Watson residual autocorrelation test ***')
        durbin = sm.stats.stattools.durbin_watson(results_df['residual'])
        print('Statistic value -',durbin)
        if durbin > 2.5:
            print('Statistic indicates negative serial correlation between residuals.')
        elif durbin > 1.5:
            print('Statistic indicates no serial correlation between residuals.')
        else:
            print('Statistic indicates positive serial correlation between residuals.')
        print()

    def homoskedasticity_plot(self):
        """
        
        """
        print('*** Investigating homoscedasticity of residuals ***')
        #Plot the model's residuals against the predicted values of y
        plt.xlabel('Predicted value')
        plt.ylabel('Residual error')
        plt.title('Model predicted values against residuals')
        plt.scatter(results_df['predicted'], results_df['residual'])
        plt.show()
        
        plt.scatter(results_df['actual'], results_df['predicted'], alpha=.35)
        plt.title("Plot of Linear Regression Test Values, Predicted vs. Actual")
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.plot([min(results_df['actual'].min(), results_df['predicted'].min()),
                max(results_df['actual'].max(), results_df['predicted'].max())],
                [min(results_df['actual'].min(), results_df['predicted'].min()),
                max(results_df['actual'].max(), results_df['predicted'].max())], color='black')
        plt.show()
    
    def w_test_homoskedasticity(self):
        """
        
        """
        print('*** White test for Homoscedasticity ***')
        #perform White's test
        # H0 : Homoscedasticity is present (residuals are equally scattered)
        # HA : Heteroscedasticity is present (residuals are not equally scattered)
        white_test = smd.het_white(results_df['residual'], X_test[scoring_vars])

        #print results of White's test
        labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
        print(lzip(labels, white_test))

        if white_test[1] > self.alpha:
            print('On', self.alpha,'significance level we fail to reject H0 about homoscedasticity of residuals.')
        else:
            print('On', self.alpha,'significance level we reject H0 and assume heteroscedasticity of residuals.')
        print()

    def bp_test_homoskedasticity(self):
        """
        
        """
        print('*** Breusch-Pagan homoscedasticity test***')
        name = ['Lagrange multiplier statistic', 'p-value', 
            'f-value', 'f p-value']
        bpagan_test = smd.het_breuschpagan(results_df['residual'], X_test[scoring_vars])
        print(lzip(name, bpagan_test))

        if bpagan_test[1] > self.alpha:
            print('On', self.alpha,'significance level we fail to reject H0 about homoscedasticity of residuals.')
        else:
            print('On', self.alpha,'significance level we reject H0 and assume heteroscedasticity of residuals.')
        print()
        # Either one can transform the variables to improve the model, or use a robust regression method
        # that accounts for the heteroscedasticity. 
    
    def vif(self):
        """
        
        """
    # Multicollinearity check
    print('*** Investigating Multicollinearity ***')
    print('*** Variance Inflation Factor (VIF) table***')
    vif_data = pd.DataFrame()
    vif_data["feature"] = data_in[scoring_dict[model_list[model_in]['variables']]].columns
    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(data_in[scoring_dict[model_list[model_in]['variables']]].values, i)
                          for i in range(len(data_in[scoring_dict[model_list[model_in]['variables']]].columns))]
    display(HTML(vif_data.to_html()))
    if vif_data["VIF"].max() > 5:
        print('There is indication (VIF > 5) that multicollinearity is present in the data.')
    else:
        print('Based on VIF there seems that there are no significant correlations between independent variables.')
    print()    

    def corr_matrix(self):
        """
        
        """
        print('*** Correlation Matrix ***')
        f, ax = plt.subplots(figsize=(40, 30))
        mat = round(data_in[scoring_dict[model_list[model_in]['variables']]].corr('pearson'),2)
        mask = np.triu(np.ones_like(mat, dtype=bool))
        cmap = sn.diverging_palette(230, 20, as_cmap=True)
        sn.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0, annot = True,
                square=True, linewidths=.5, cbar_kws={"shrink": .7})
        plt.show()
