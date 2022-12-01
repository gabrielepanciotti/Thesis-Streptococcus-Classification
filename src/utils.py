import re


def remove_pharentesis(column: str) -> str:
    """Rename iris dataset columns removing unit of measure inside pharentesis.
    
    Parameters
    ----------
    column : str
        column to process
        
    Returns
    -------
    str
        column renamed
    
    """
    return re.sub(pattern=r"\s\(.*\)", repl="", string=column)