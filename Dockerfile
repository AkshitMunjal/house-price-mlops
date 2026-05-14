# lightweight python image
FROM python:3.11-slim

# set working directory
WORKDIR /app

# copy requirements file
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# create required folders
RUN mkdir -p src/training

# copy frontend
COPY frontend ./frontend

# copy api package
COPY src/api ./src/api

# copy feature engineering
COPY src/features ./src/features

# copy required preprocessing module
COPY src/training/preprocessing.py ./src/training/preprocessing.py

# copy root package init
COPY src/__init__.py ./src/__init__.py

# expose FastAPI port
EXPOSE 8000

# run FastAPI application
CMD ["uvicorn", "src.api.main_api:app", "--host", "0.0.0.0", "--port", "8000"]


# use this command to build the docker image:
# -> docker build -t house-price-app:latest .

# use this command to run the docker container after building the image:
# -> docker run -p 8000:8000 --env-file .env house-price-app:latest