FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for GeoPandas
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate synthetic data for demo mode
RUN python -c "from src.data_prep import prepare_all_data; prepare_all_data(use_osm=False)"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "webapp/streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
