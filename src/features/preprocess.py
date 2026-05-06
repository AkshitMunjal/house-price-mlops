from .feature_config import (IMPORTANT_COLUMNS, 
                            THRESHOLD,
                            CRITICAL_COLUMNS,
                            NUMERICAL_COLUMNS,
                            CATEGORICAL_COLUMNS)
import pandas as pd


# Dynamic function to drop columns with a high percentage of null values, 
# while ensuring important columns are retained.
def drop_high_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with a high percentage of null values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with high null columns dropped.
    """
    # Calculate the percentage of null values in each column
    null_percentages = df.isnull().mean()

    # Identify columns to drop based on the threshold
    columns_to_drop = null_percentages[null_percentages > THRESHOLD].index

    # Ensure important columns are not dropped
    columns_to_drop = [col for col in columns_to_drop if col not in IMPORTANT_COLUMNS]

    # Drop the identified columns
    df_dropped = df.drop(columns=columns_to_drop)

    return df_dropped


def validate_input(df: pd.DataFrame) -> None:

    """"
    This checks if the critical columns are present in the dataframe

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Raises:
        ValueError: If any of the critical columns are missing from the DataFrame.

    Returns:
        This function does not return anything. It raises an error if validation fails.
    """

    required_columns = CRITICAL_COLUMNS + IMPORTANT_COLUMNS

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame")
        

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with missing values handled.
    """

    # Drop rows with missing values in critical columns
    # we are running for loop inside subset to check if the critical columns are present in the dataframe 
    # before dropping rows with null values in those columns. This ensures that we only attempt to drop 
    # rows based on columns that actually exist in the DataFrame, preventing potential errors.
    df = df.dropna(subset=[col for col in CRITICAL_COLUMNS if col in df.columns]).copy()


    # Fill numerical columns with median
    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(df[col].median())

    # Fill categorical columns with mode
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(df[col].mode()[0])

    return df

def validate_no_critical_nulls(df: pd.DataFrame) -> None:
    """
    Validate that there are no null values in critical columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Raises:
    ValueError: If any critical column contains null values.
    """

    for col in CRITICAL_COLUMNS:
        if col in df.columns and df[col].isnull().sum() > 0:
            raise ValueError(f"Critical column '{col}' contains null values after handling missing values.")
        
def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for duplicate rows in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing only the duplicate rows.
    """
    duplicates = df[df.duplicated(keep=False)]
    return duplicates

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows from the DataFrame.
    Two houses will be duplicate if they have same location, size, total_sqft, and price. 
    not necessarily if they have same bath and balcony.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with duplicate rows dropped.
    """
    DUPLICATE_SUBSET = ['location', 'size', 'total_sqft', 'price']
    df_duplicates = df.drop_duplicates(subset=DUPLICATE_SUBSET)
    return df_duplicates

# Main function to preprocess the data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by dropping high null columns, validating input, handling missing values, and validating critical columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """

    # Step 1: Validate input DataFrame
    validate_input(df)

    # Step 2: Drop columns with high percentage of null values
    df = drop_high_null_columns(df)

    # Step 3: Handle missing values
    df = handle_missing_values(df)

    # Step 4: Check and drop duplicate rows
    if check_duplicates(df).shape[0] > 0:
        df = drop_duplicates(df)

    # Step 5: Validate that there are no null values in critical columns
    validate_no_critical_nulls(df)

    return df