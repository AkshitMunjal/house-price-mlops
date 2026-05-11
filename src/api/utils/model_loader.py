import os

from dotenv import load_dotenv

import dagshub
import mlflow
import mlflow.xgboost
import joblib
from mlflow.artifacts import download_artifacts

# load environment variables
load_dotenv()


class ModelLoader:

    # cache loaded model
    _model = None

    # cache known locations
    _known_locations = None

    # prevent repeated dagshub initialization
    _dagshub_initialized = False

    @classmethod
    def _init_dagshub(cls):

        # avoid repeated initialization
        if cls._dagshub_initialized:
            return

        # validate token exists
        if not os.getenv("DAGSHUB_USER_TOKEN"):
            raise RuntimeError(
                "DAGSHUB_USER_TOKEN is missing."
            )

        # initialize dagshub + mlflow
        dagshub.init(
            repo_owner="AkshitMunjal",
            repo_name="house-price-mlops",
            mlflow=True
        )

        cls._dagshub_initialized = True

    @classmethod
    def load_model(cls):

        # return cached model if already loaded
        if cls._model is not None:
            return cls._model

        # initialize dagshub
        cls._init_dagshub()

        # load production model from mlflow registry
        cls._model = mlflow.xgboost.load_model(
            "models:/HousePriceModel@production"
        )

        return cls._model

    
    @classmethod
    def load_known_locations(cls):
        
        # return cached known locations if already loaded
        if cls._known_locations is not None:
            return cls._known_locations

        # initialize dagshub
        cls._init_dagshub()

        # download artifacts locally
        artifact_path = download_artifacts(
            artifact_uri="models:/HousePriceModel@production/"
            "known_locations/known_locations.joblib"
        )
        

        # load known locations list from disk
        cls._known_locations = joblib.load(artifact_path)

        return cls._known_locations



if __name__ == "__main__":

    test = ModelLoader.load_model()

    print(f"Model loaded successfully: {test}")


# python src/api/utils/model_loader.py