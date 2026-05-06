from .csv_loader import CSVDataHandler
import pandas as pd

HANDLERS = {
    'csv': CSVDataHandler(),
    # Future handlers can be added here (e.g., 'parquet': ParquetDataHandler())
    # 'json': JSONDataHandler(),
    # "s3": S3DataHandler(),
    # "sql": SQLDataHandler()
}

def load_raw_data(source: str, source_type: str = 'csv') -> pd.DataFrame:
    """
    Loads data from a specified source using the appropriate data handler.
    
    Parameters:
        source (str): The source from which to load the data (e.g., file path, database connection string)
        source_type (str): The type of the source (e.g., 'csv', 'parquet', 'database')

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data
    """
    # For now, we only support CSV files, but this can be extended to support other formats

    if source_type not in HANDLERS:
        raise ValueError(f"Unsupported data format for source: {source_type}. Currently only CSV files are supported.")

    handler = HANDLERS[source_type]
    return handler.load_data(source)