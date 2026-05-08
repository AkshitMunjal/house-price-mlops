import os
import boto3
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

from .base import BaseDataHandler
from .csv_loader import CSVDataHandler


load_dotenv()


class S3DataHandler(BaseDataHandler):

    """
    Data handler for downloading data from Amazon S3,
    saving it locally, and loading it using CSVDataHandler.
    """

    def __init__(self):

        # CHANGED:
        # loading AWS credentials from .env file
        # so boto3 can authenticate with AWS S3
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION")


        # creating boto3 S3 client
        # this client communicates with AWS S3
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region
        )

    def load_data(self, source: str) -> pd.DataFrame:

        """
        Downloads a CSV file from S3,
        saves it locally inside data/raw/,
        then loads it using CSVDataHandler.

        Parameters:
            source (str): S3 URI
            Example:
                s3://bucket-name/file.csv

        Returns:
            pd.DataFrame: Loaded dataset
        """

        # validating S3 URI format before processing
        if not source.startswith("s3://"):

            raise ValueError(
                "Invalid S3 URI format. Expected format: s3://bucket/key"
            )


        # removing s3:// prefix
        # because boto3 expects bucket/key separately
        s3_path = source.replace("s3://", "")


        # splitting bucket name and file path
        # example:
        # house-price-training-data-akshit/bengaluru_house_prices.csv
        # becomes:
        # bucket_name = house-price-training-data-akshit
        # key = bengaluru_house_prices.csv
        bucket_name, key = s3_path.split("/", 1)


        # defining local download location
        # downloaded file will overwrite/create:
        # data/raw/bengaluru_house_prices.csv
        BASE_DIR = Path(__file__).resolve().parents[2]
        local_path = (
            BASE_DIR /
            "data" /
            "raw" /
            Path(key).name
        )


        # automatically creating data/raw directory
        # if it does not already exist
        local_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            # downloading file from AWS S3
            # into local filesystem
            self.s3_client.download_file(
                bucket_name,
                key,
                str(local_path)
            )

            print(f"Downloaded file from S3 to: {local_path}")

        except Exception as e:

            raise RuntimeError(
                f"Error downloading file from S3: {e}"
            ) from e

        csv_handler = CSVDataHandler()

        return csv_handler.load_data(str(local_path))