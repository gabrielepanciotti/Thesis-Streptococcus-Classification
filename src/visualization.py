import pandas
import numpy
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot


def regression_diagnostic_plot(y_true: numpy.array, y_pred: numpy.array) -> None:
    """Create diagnostic plots for regression models.
    
    Parameters
    ----------
    y_true : numpy.array
        target values
    y_pred : numpy.array
        predicted values
    
    Returns
    -------
    None
        residuals plot
    
    """
    residuals = y_true - y_pred
    xmin, xmax = y_pred.min(), y_pred.max()
    fig = plt.figure(figsize=(15, 5))
    ax = fig.subplots(nrows=1, ncols=2)
    # Residuals plot
    ax[0].set(title="Residuals vs. fitted plot", xlabel="Fitted values", ylabel="Residuals")
    ax[0].hlines(y=0, xmin=xmin, xmax=xmax, colors="red", linestyles="--", linewidth=2)
    sns.scatterplot(x=y_pred, y=residuals, ax=ax[0])
    # Q-Q plot 
    ax[1].set_title("Q-Q plot of residuals")
    qqplot(data=residuals, line="45", fit="True", markersize=5, ax=ax[1])
    plt.tight_layout()
    fig.show()


def regression_plane_plot(X: numpy.ndarray, y: numpy.array, model, **axis_params):
    """Plot regression plane for regression models.
    
    Parameters
    ----------
    X : numpy.array
        features matrix
    y : numpy.array
        target vector
    model : Any
        Any scikit-learn regressor
    axis_params
        parameters to customize axis
        reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html
    
    Returns
    -------
    None
        regression plane plot
        
    """
    if X.shape[1] != 2:
        raise ValueError("The X array must have two columns")
    else:
        # compute predictors minimum and maximum values
        xmin, xmax = X[:, 0].min(), X[:, 0].max()
        ymin, ymax = X[:, 1].min(), X[:, 1].max()
        # define plane axis values
        x = numpy.array([[xmin, xmin], [xmax, xmax]])
        Y = numpy.array([[ymin, ymax], [ymin, ymax]])
        xy = numpy.array([[xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax]]).T
        Z = model.predict(X=xy).reshape(2, 2)
        # plot of the figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection='3d')
        ax.set(**axis_params)
        ax.scatter(xs=X[:, 0], ys=X[:, 1], zs=y, c="blue")
        ax.plot_surface(X=x, Y=Y, Z=Z, alpha=0.5)
        fig.show()


def categorical_count_plot(data: pandas.DataFrame, column: str) -> None:
    """Create count plot for classification problems.
    
    Parameters
    ----------
    data : pandas.DataFrame
        input dataframe
    column : str
        categorical column to plot
    
    Returns
    -------
    None
        count plot
    
    """
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x=column, data=data)
    ax.bar_label(ax.containers[0], padding=1)
    plt.title(label=f"{column} barplot")
    plt.tight_layout()
    plt.show()
    
    
def feature_importances_plot(model, labels: list, **kwargs) -> None:
    """Compute normalized feature importance from model and returns the data or show plot.
    
    Parameters
    ----------
    model : Any
        scikit-learn model
    labels : list
        list of feature labels
    
    Returns
    -------
    None
        feature importances plot
    
    """
    feature_importances = model.feature_importances_
    feature_importances = 100 * (feature_importances / feature_importances.max())
    series = pandas.Series(data=feature_importances, index=labels).sort_values()
    series.plot(kind="barh", figsize=(5, 7), title=f"Feature importances", legend=None, **kwargs)
    plt.tight_layout()
    plt.show()
