import pandas as pd
from src.api.utils.monitoring import generate_monitoring_flags


def test_generate_monitoring_flags():

    df = pd.DataFrame({
        'sqft_per_bhk': [1600],
        'bath': [8],
        'new_total_sqft': [4000],
        'original_location': ['Whitefield']
    })

    flags = generate_monitoring_flags(df)

    assert flags['high_sqft_per_bhk'] == True
    assert flags['low_sqft_per_bhk'] == False


# Use the following Command to run the test:
# pytest tests/test_monitoring.py -v