FROM python:3.11-slim

# System dependencies needed by open3d and opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies (separate layer for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gdown

# Copy source
COPY pipeline/ pipeline/
COPY sim_config.yaml .

# Download and unzip data from Google Drive
RUN gdown "https://drive.google.com/uc?id=1EPeMx8w3NBVmJz1cKLztCbOE9v25WPP3" -O /tmp/conveyor-sim.zip \
    && unzip -q /tmp/conveyor-sim.zip -d /workspace/input \
    && rm /tmp/conveyor-sim.zip