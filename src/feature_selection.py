import numpy


def variance_inflation_factor(X: numpy.ndarray) -> numpy.ndarray:
    """Compute the predictor's variance inflation factor.
    
    Parameters
    ----------
    X : numpy.ndarray
        matrix with predictors
        
    Returns
    -------
    numpy.ndarray
        vif for each predictor
    
    """
    pcc = numpy.corrcoef(x=X, rowvar=False)
    vif = numpy.linalg.inv(a=pcc)
    return vif.diagonal()
