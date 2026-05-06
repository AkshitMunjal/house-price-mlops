import pandas as pd


# one hot encoding for area_type, location columns as they are categorical columns
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:

    """
    this function is applied on categorical features to convert them into numerical features using one hot encoding

    Parameters:
        df (pd.DataFrame): input dataframe containing categorical features

    Returns:
        pd.DataFrame: dataframe with one hot encoded features
    """

    # area_type encoding
    area_type_dummies = pd.get_dummies(df['area_type'], drop_first=True)
    df = pd.concat([df, area_type_dummies], axis=1)
    df = df.drop(columns=['area_type'])

    # location encoding
    location_dummies = pd.get_dummies(df['location'], drop_first=True)
    df = pd.concat([df, location_dummies], axis=1)
    df = df.drop(columns=['location'])

    return df

# column selection based on VIF & RCA
def select_features(df: pd.DataFrame) -> pd.DataFrame:

    """
    this function is applied on the dataframe to select features based on VIF and RCA analysis

    Parameters:
        df (pd.DataFrame): input dataframe containing all features

    Returns:
        pd.DataFrame: dataframe with selected features
    """

    # based on VIF and RCA analysis, we are selecting the following features
    drop_cols = ['bath','new_size','sqft_per_bhk']
    df = df.drop(columns=drop_cols)

    return df


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    """
    this function is applied on the dataframe to split features and target variable

    Parameters:
        df (pd.DataFrame): input dataframe containing all features and target variable

    Returns:
        X (pd.DataFrame): dataframe containing features
        y (pd.Series): series containing target variable
    """

    target_col = 'price'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


# main function for full preprocessing pipeline
def preprocess_for_training(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    """
    this function is the main function for the full preprocessing pipeline which includes encoding categorical features, selecting features based on VIF and RCA analysis, and splitting features and target variable

    Parameters:
        df (pd.DataFrame): input dataframe containing all features and target variable

    Returns:
        X (pd.DataFrame): dataframe containing features
        y (pd.Series): series containing target variable
    """

    df_encoded = encode_categorical_features(df)
    df_selected = select_features(df_encoded)
    X, y = split_features_and_target(df_selected)

    return X, y