# Use a lightweight Python version
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

# Upgrade pip and remove conflicting Google packages
RUN pip install --upgrade pip && \
    pip uninstall -y google google-ai-generativelanguage || true && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port for Cloud Run
EXPOSE 8080

# Run Streamlit on Cloud Run
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
