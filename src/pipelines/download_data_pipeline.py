import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.load_data import load_raw_data

# raw_data_path = BASE_DIR / 'data' / 'raw' / 'bengaluru_house_prices.csv'
S3_DATA_PATH = (
    "s3://house-price-training-data-akshit/"
    "bengaluru_house_prices.csv"
)

# df = load_raw_data(raw_data_path, source_type='csv')
df = load_raw_data(source=S3_DATA_PATH, source_type='s3')

print("S3 data download pipeline completed successfully")