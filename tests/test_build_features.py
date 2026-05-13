import pandas as pd
from src.features.build_features import (convert_total_sqft_to_numeric,
                                         transform_size_column,
                                         apply_domain_filters,
                                         build_features)


def test_convert_total_sqft_range():

    result = convert_total_sqft_to_numeric('2100-2850')
    
    assert result == 2475.0

def test_convert_total_sqft_single_value():

    result = convert_total_sqft_to_numeric('1200')
    
    assert result == 1200.0

def test_convert_total_sqft_invalid_value():

    result = convert_total_sqft_to_numeric('abc')
    
    assert pd.isna(result)

def test_transform_size_column():

    df = pd.DataFrame({'size': ['2 BHK', '3 BHK', '4 BHK', '1 RK']})
    
    transformed_df = transform_size_column(df)
    
    expected_bedrooms = [2.0, 3.0, 4.0, 1.0]
    
    assert transformed_df['new_size'].tolist() == expected_bedrooms
    assert 'size' not in transformed_df.columns

def test_apply_domain_filters():

    df = pd.DataFrame({
        'new_size' : [2,12],
        'new_total_sqft' : [1200, 8000],
        'bath' : [2,10]
    })
    
    filtered_df = apply_domain_filters(df)
    
    # it means that only the first row is valid and the second row is filtered out due to unrealistic values
    assert len(filtered_df) == 1

    assert filtered_df.iloc[0]['new_size'] == 2
    assert filtered_df.iloc[0]['new_total_sqft'] == 1200
    assert filtered_df.iloc[0]['bath'] == 2

def test_build_features_pipeline():

    df = pd.DataFrame({
        'total_sqft': ['1200', '1500'],
        'size': ['2 BHK', '3 BHK'],
        'bath': [2, 3],
        'price': [50, 75],
        'availability': ['Ready To Move', 'Under Construction'],
        'location': ['Whitefield', 'Whitefield']
    })

    transformed_df = build_features(df)

    assert 'new_total_sqft' in transformed_df.columns
    assert 'new_size' in transformed_df.columns
    assert 'price_per_sqft' not in transformed_df.columns
    assert 'total_sqft' not in transformed_df.columns
    assert 'size' not in transformed_df.columns

    # this will ensure that the transformed DataFrame is not empty after applying the feature engineering 
    assert len(transformed_df) > 0



#  run using the following commands in the terminal from the root directory of the project:
#  pytest tests/test_build_features.py -v