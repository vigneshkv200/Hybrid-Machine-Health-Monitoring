FROM python:3.10

# Set working folder inside container
WORKDIR /app

# Copy everything into /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose streamlit port
EXPOSE 8501

# Start Streamlit using correct file name
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
