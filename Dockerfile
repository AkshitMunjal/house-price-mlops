# lightweight python image
FROM python:3.11-slim

# set working directory
WORKDIR /app

# copy requirements file
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# create required package folders
RUN mkdir -p src/features
RUN mkdir -p src/training

# copy frontend
COPY frontend ./frontend

# copy api package
COPY src/api ./src/api

# copy required feature engineering file
COPY src/features/build_features.py \
./src/features/build_features.py

# copy required preprocessing file
COPY src/training/preprocessing.py \
./src/training/preprocessing.py

# copy root package init file
COPY src/__init__.py ./src/__init__.py

# expose FastAPI port
EXPOSE 8000

# run FastAPI application
CMD ["uvicorn", "src.api.main_api:app", "--host", "0.0.0.0", "--port", "8000"]


# use this command to run the docker container after building the image:
# docker run -p 8000:8000 --env-file .env house-price-app:latest