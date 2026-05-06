import pandas as pd
from pathlib import Path
from .base import BaseDataHandler


class CSVDataHandler(BaseDataHandler):

    """
    Data handler for loading CSV files
    Inherits from BaseDataHandler and implements the load_data method
    """

    def load_data(self, source: str) -> pd.DataFrame:
        """
        Loads data from a CSV file and returns it as a pandas DataFrame.
        
        Parameters:
            source (str): The file path to the CSV file

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data
        """
        # Check if the file exists
        if not Path(source).is_file():
            raise FileNotFoundError(f"The file {source} does not exist.")
        
        # Load the CSV file into a DataFrame
        try:
            df = pd.read_csv(source)
            return df
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the CSV file: {e}") from e