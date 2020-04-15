def datetimeFormat(ns: bool = False) -> str:
    """Get the datetime format format for datetime strptime and strftime

    Returns
    -------
    str
        The datetime str format
    """
    if ns:
        return "%Y-%m-%d %H:%M:%S.%f"
    return "%Y-%m-%d %H:%M:%S"