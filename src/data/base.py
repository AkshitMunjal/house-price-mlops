from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from typing import Union

# Base class for data handling
# This is designed for the data ingestion layer to support multiple sources 
# like CSV, S3, and SQL using a factory pattern

class BaseDataHandler(ABC):

    """
    Abstract base class for data loading
    All data handlers should inherit from this class and 
    implement the load_data method
    this will return a pandas DataFrame containing the loaded data
    """

    @abstractmethod
    def load_data(self, source: Union[str, Path]) -> pd.DataFrame:
        
        """
        This function takes a source (e.g., file path, database connection string) 
        and returns a pandas DataFrame containing the loaded data.

        Parameters:
            source (Union[str, Path]): The source from which to load the data (e.g., file path, database connection string)
        Returns:
            pd.DataFrame: A DataFrame containing the loaded data
        """
        pass