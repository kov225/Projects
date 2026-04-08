FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PyMC (gcc, g++, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add project code
COPY . .

# Run the simulator by default if needed, 
# but usually we want to run notebooks or the optimizer.
CMD ["python", "simulator.py"]
