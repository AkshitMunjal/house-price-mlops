import pandas as pd

# monitoring thresholds as per the features/build_features.py file
HIGH_SQFT_PER_BHK = 1500
LOW_SQFT_PER_BHK = 350
HIGH_BATH_COUNT = 7
HIGH_TOTAL_SQFT = 6000
LOW_TOTAL_SQFT = 400

def generate_monitoring_flags(df: pd.DataFrame) -> dict:

        """
        Generate monitoring flags based on the input DataFrame
        This function checks for certain conditions in the input 
        data and generates flags that can be used for monitoring purposes
        For example: 6000 sqft for 1 BHK is an outlier and can be flagged for monitoring and
        400 sqft for 4 BHK is also an outlier and can be flagged for monitoring

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the features

        Returns:
            dict: A dictionary containing the monitoring flags
        """
        # We are doing this because prediction will be made on a single row of data, so we can 
        # safely take the first row of the DataFrame to check the conditions for monitoring flags
        row = df.iloc[0]

        flags = {
            'high_sqft_per_bhk': bool(row['sqft_per_bhk'] > HIGH_SQFT_PER_BHK),
            'low_sqft_per_bhk': bool(row['sqft_per_bhk'] < LOW_SQFT_PER_BHK),
            'high_bath_count': bool(row['bath'] > HIGH_BATH_COUNT),
            'high_total_sqft': bool(row['new_total_sqft'] > HIGH_TOTAL_SQFT),
            'low_total_sqft': bool(row['new_total_sqft'] < LOW_TOTAL_SQFT),
            'unseen_location': bool(row['location'] == 'Other')

        }

        return flags