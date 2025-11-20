# Use official Python image (supports TensorFlow 2.12)
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y libhdf5-dev libglib2.0-0

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "Advanced_Hybrid_ML_Project/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
