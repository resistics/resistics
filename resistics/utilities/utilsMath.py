from typing import Union

# import from package
from resistics.utilities.utilsPrint import errorPrint

def intdiv(nom: Union[int, float], div: Union[int, float]) -> int:
    """Return an integer result of division

    The division is expected to be exact and ensures an integer return rather than float.
    Code execution will exit if division is not exact
    
    Parameters
    ----------
    nom : int, float
        Nominator
    div : int, float
        Divisor    

    Returns
    -------
    out : int
        Result of division
    """

    if nom % div == 0:
        return nom // div
    else:
        errorPrint(
            "utilsMath::intdiv",
            "intdiv assumes exits upon having a remainder to make sure errors are not propagated through the code",
            quitRun=True,
        )
        return 0
