"""Base resistics classes used throughout the package"""
from typing import List, Dict, Any


class ResisticsBase(object):
    """Resistics base class

    Parent class to ensure consistency of print methods

    Methods
    -------
    __repr__()
        Print status information
    __str__()
        Convert object to a string
    info()
        Return a list of information to print out
    """

    def __repr__(self) -> str:
        """Print class information"""
        return "\n".join(self.to_string())

    def __str__(self) -> str:
        """Print class information"""
        return self.__repr__()

    def to_string(self) -> List[str]:
        raise NotImplementedError("This should be implemented in child classes")


class ProcessRecord(ResisticsBase):
    def __init__(
        self, process_name: str, parameters: Dict[str, Any], comment: str = None
    ) -> None:
        self._name = process_name
        self._parameters = parameters
        self._comment = comment

    def to_dict(self) -> Dict[str, Any]:
        """Return the record as a dictionary

        Returns
        -------
        Dict[str, Any]
            Record of process as a dictionary
        """
        return {
            "name": self._name,
            "parameters": self._parameters,
            "comment": self._comment,
        }
    
    def to_string(self) -> str:
        """Represent the process record as a string

        Returns
        -------
        str
            Process record as a string
        """
        return str(self.to_dict())


class History(ResisticsBase):
    """Class for storing processing history"""

    def __init__(self, json_str):
        self._records: List[ProcessRecord] = []
