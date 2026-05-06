import pandas as pd


def convert_total_sqft_to_numeric(x):
    """
    Convert the 'total_sqft' column to numeric values. 
    If the value is a range (e.g., '2100 - 2850'), it will take the average of the two numbers.
    If the value cannot be converted to a number, it will return NaN.

    Parameters:
        x (str): The input value from the 'total_sqft' column.

    Returns:
        float: The numeric value of 'total_sqft' or NaN if conversion fails.
    """
    try:
        if isinstance(x, str):
            if '-' in x:
                tokens = x.split('-')
                if len(tokens) == 2:
                    return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2
            return float(x.strip())
        elif isinstance(x, (int, float)):
            return float(x)
        else:
            return pd.NA
    except ValueError:
        return pd.NA

def transform_total_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the 'convert_total_sqft_to_numeric' function to the 'total_sqft' column of the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'new_total_sqft' column transformed to numeric values.
    """
    df['new_total_sqft'] = df['total_sqft'].apply(convert_total_sqft_to_numeric)

    # change dtype from object to float64
    df['new_total_sqft'] = pd.to_numeric(df['new_total_sqft'], errors='coerce')

    # drop the original 'total_sqft' column
    df = df.drop(columns=['total_sqft'])

    # drop rows where 'new_total_sqft' is null
    df = df.dropna(subset=['new_total_sqft']).copy()

    return df

def transform_size_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the 'size' column to extract the number of bedrooms as a numeric value.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'new_size' column transformed to numeric values.
    """

    # extract numeric part from the 'size' column and convert it to float (e.g., '2 BHK' -> 2.0)
    df['new_size'] = df['size'].str.extract(r'(\d+)').astype(float)

    # drop the original 'size' column
    df = df.drop(columns=['size'])

    # drop the rows where 'new_size' is null
    df = df.dropna(subset=['new_size']).copy()

    return df

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features based on existing columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new derived features added.
    """
    # this creates a new feature 'price_per_sqft' which is the price per square foot, calculated as (price * 100000) / new_total_sqft
    df['price_per_sqft'] = (df['price'] * 100000) / df['new_total_sqft']

    # this creates a new feature 'sqft_per_bhk' which is the square footage per bedroom, calculated as new_total_sqft / new_size
    df['sqft_per_bhk'] = df['new_total_sqft'] / df['new_size']
    
    # this creates a new feature 'bath_per_bhk' which is the number of bathrooms per bedroom, calculated as bath / new_size
    df['bath_per_bhk'] = df['bath'] / df['new_size']

    return df

def apply_domain_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-specific filters to remove unrealistic values from the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with unrealistic values filtered out.
    """
    
    # filter out rows in new_size column where bhk is greater than 9 bhk    
    df = df[df['new_size'] <= 9].copy()

    # filter out rows in new_total_sqft column where total_sqft is less than 400 sqft or greater than 6000 sqft.
    df = df[(df['new_total_sqft'] >= 400)& (df['new_total_sqft'] <= 6000)].copy()

    # filter out rows in bath column where bath is greater than 7
    df = df[df['bath'] <= 7].copy()

    return df

def remove_price_per_sqft_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from the DataFrame based on the 'price_per_sqft' column using local filtering by location.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with 'price_per_sqft' outliers removed.
    """
    # this function removes outliers from the DataFrame based on the 'price_per_sqft' column using local filtering by location.
    
    df_out = pd.DataFrame()

    for key, subdf in df.groupby('location'):
        m = subdf['price_per_sqft'].mean()
        st = subdf['price_per_sqft'].std()
        reduced_df = subdf[(subdf['price_per_sqft'] >= (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)

    return df_out

def apply_feature_filters(df: pd.DataFrame) -> pd.DataFrame:

    """
    this function applies filters based on the derived features 'sqft_per_bhk' and 'bath_per_bhk' to remove unrealistic values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with unrealistic values filtered out based on 'sqft_per_bhk' and 'bath_per_bhk'.

    """

    df = df[(df['sqft_per_bhk'] >= 350) & ((df['sqft_per_bhk'] <= 1500))].copy()
    df = df[(df['bath_per_bhk'] <= 2) & ((df['bath_per_bhk'] >= 0.5))].copy()

    return df

def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    
    
    """"
    this function finalizes the features by dropping the 'price_per_sqft' column if it exists in the DataFrame.
    we are dropping price_per_sqft to prevent data leakage because price_per_sqft is derived from the target variable 'price' and the feature 'new_total_sqft'.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'price_per_sqft' column dropped if it exists.
    
    """

    if 'price_per_sqft' in df.columns:
        df = df.drop(columns=['price_per_sqft'])

    return df

def apply_price_per_sqft_cap(df: pd.DataFrame) -> pd.DataFrame:

    """"
    this function applies a cap to the 'price_per_sqft' column by filtering out rows where 'price_per_sqft' is greater than 20000.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with rows where 'price_per_sqft' is greater than 20000 filtered out.
    """

    df = df[df['price_per_sqft']<20000].copy()
    return df

def handle_text_based_features(df: pd.DataFrame) -> pd.DataFrame:

    """
    This function handles text-based features in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with text-based features handled appropriately.
    """
    
    # availability column: this column contains text values such as 'Ready To Move', 'Immediate Possession', 'Under Construction', etc. We will convert these text values into binary values where 'Ready To Move' and 'Immediate Possession' will be represented as 1 (indicating that the property is ready for occupancy), and all other values will be represented as 0 (indicating that the property is not ready for occupancy).
    df['availability'] = df['availability'].apply(lambda x: 1 if x == 'Ready To Move' or x == 'Immediate Possession' else 0)

    # location grouping
    location_counts = df['location'].value_counts()
    threshold = 15

    df['location'] = df['location'].apply(lambda x: x if location_counts[x] >= threshold else 'Other')

    return df

# main function to build features
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for the input DataFrame by transforming the 'total_sqft', 'size' columns,
    build new features such as 'price_per_sqft', 'sqft_per_bhk', and 'bath_per_bhk', 
    and filter unrealistic values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with built features.
    """

    df = transform_total_sqft(df)
    df = transform_size_column(df)
    df = create_derived_features(df)
    df = apply_domain_filters(df)
    df = remove_price_per_sqft_outliers(df)
    df = apply_price_per_sqft_cap(df)
    df = apply_feature_filters(df)
    df = handle_text_based_features(df)
    df = finalize_features(df)
    return df