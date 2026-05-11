import os
import json
import uuid

from io import BytesIO
from datetime import datetime, timezone

import boto3
from dotenv import load_dotenv


load_dotenv()


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)


def upload_prediction_log(input_data, prediction,monitoring_flags):

    """
    Upload a prediction log to S3. This function takes the input data, the predicted price, 
    and the monitoring flags, and uploads them as a JSON file to an S3 bucket.

    Stores:
        - timestamp: The time when the prediction was made
        - input: The raw input data that was used for the prediction
        - prediction: The predicted price in lakhs
        - monitoring_flags: The flags generated for monitoring based on the input data

    """


    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": prediction,
        "monitoring_flags": monitoring_flags
    }

    file_name = f"prediction_logs/{uuid.uuid4()}.json"

    json_bytes = json.dumps(
        log_data,
        indent=2
    ).encode("utf-8")

    s3_client.upload_fileobj(
        Fileobj=BytesIO(json_bytes),
        Bucket=S3_BUCKET_NAME,
        Key=file_name
    )

    print(f"S3 log uploaded: {file_name}")