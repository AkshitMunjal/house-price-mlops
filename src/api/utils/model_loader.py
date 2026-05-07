import mlflow
import mlflow.xgboost
import dagshub


# connect to dagshub to access the mlruns directory where the mlflow models are stored
dagshub.init(
    repo_owner="AkshitMunjal",
    repo_name="house-price-mlops",
    mlflow=True
)

class ModelLoader:

    # this is a class variable to hold the loaded model
    # it is called model caching to avoid loading the model multiple times which can be expensive
    # at first request the model will be loaded and stored in this variable, subsequent requests will use the cached model
    _model = None


    # using @classmethod decorator to define a class method that can be called on the class itself without needing an instance of the class
    # for example without this: we use obj = ModelLoader(), obj.load_model() to load the model, but with @classmethod we can directly call 
    # ModelLoader.load_model() to load the model without creating an instance of the class
    @classmethod
    def load_model(cls):

        # if the model is already loaded and cached in the class variable _model, 
        # return it directly to avoid loading it again
        if cls._model is not None:
            return cls._model
        

        # using mlflow model registry model that is in production stage and version 1 to load the model
        cls._model = mlflow.xgboost.load_model(
            "models:/HousePriceModel@production"
            )
        
        return cls._model
    
if __name__ == "__main__":
    test = ModelLoader.load_model()
    print(f"Model loaded successfully: {test}")


# python src/api/utils/model_loader.py