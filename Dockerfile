# Base image with GCC and Python
FROM ubuntu:20.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      g++ \
      cmake \
      python3 \
      python3-pip \
      && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and build C++ code
COPY src/ src/
COPY include/ include/
RUN mkdir -p build && \
    g++ -std=c++17 -Wall -O2 src/*.cpp -Iinclude -o bin/vehicle_safety_classifier

# Copy Python requirements (for tests or plotting)
COPY requirements-dev.txt .
RUN pip3 install --no-cache-dir -r requirements-dev.txt

# Copy configs and scripts
COPY config/ config/
COPY bin/ bin/        # If you pre-built, else build here
COPY results/ results/

# Expose any ports if you have a server component (e.g., 8000)
# EXPOSE 8000

# Default entrypoint: run the classifier with a dev config
ENTRYPOINT ["./bin/vehicle_safety_classifier", "--config", "config/dev.yaml"]
