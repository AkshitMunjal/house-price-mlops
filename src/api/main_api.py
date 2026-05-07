import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI
from src.api.routes.prediction_routes import router as prediction_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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


# mount the frontend directory to serve static files
app.mount(
    "/frontend",
    StaticFiles(directory="frontend"),
    name="frontend",
)

# serve index.html at the root endpoint
@app.get("/")
def server_endpoint():
    return FileResponse("frontend/index.html")


# use this command to run the app:
# uvicorn src.api.main_api:app --reload