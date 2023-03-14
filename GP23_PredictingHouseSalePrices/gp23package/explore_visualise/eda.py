import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from IPython.display import display, HTML


def var_boxplot(var, data, target, title, xrotate=0):
    """ Create boxplot of variable vs descending median of target
    and provide frequency table for variable values. Caution:
    target variable has to be numeric.

    Required libraries:
    import matplotlib.pyplot as plt
    import seaborn as sn
    import pandas as pd
    from IPython.display import display, HTML

    - **parameters**, **types**, **return** and **return types**::

    :param data: DataFrame to plot and calculate frequencies
    :type data: str
    :param var: Variable name for analysis. Provide in quotation marks.
    :type var: str
    :param target: Target variable name for analysis. Provide in
    quotation marks.
    :type target: str
    :param title: Plot title
    :type title: str
    :param xrotate: Rotation parameter for plt.xticks. By default = 0.
    :type xrotate: int
    :return: Boxplot and frequency table
    :rtype: Matplotlib plot
    """
    plt.figure(figsize=(10, 5))
    sn.boxplot(x=var, y=target, data=data)
    plt.title(title)
    plt.xlabel(var)
    plt.ylabel('Median of '+str(target))
    plt.xticks(rotation=xrotate)
    plt.show()

    valueCounts = pd.concat([data[var].value_counts(),
                 data[var].value_counts(normalize=True).mul(100)], axis=1,
                 keys=('counts', 'percentage'))
    print(var, '\n', len(data[var].unique()), 'unique values: ')
    display(HTML(valueCounts.to_html()))
