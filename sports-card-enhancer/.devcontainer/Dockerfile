FROM mcr.microsoft.com/devcontainers/python:1-3.10

# Install system dependencies for OpenCV and image processing
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libopenjp2-7 \
        libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements files
COPY backend/requirements.txt requirements-streamlit.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-streamlit.txt

# Create non-root user (devcontainers base image already has vscode user)
# The base image handles user creation

# Default command
CMD ["sleep", "infinity"]
