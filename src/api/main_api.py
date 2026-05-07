import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI
from src.api.routes.prediction_routes import router as prediction_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ML API",
    description="API for machine learning predictions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)

# use this command to run the app:
# uvicorn src.api.main_api:app --reload