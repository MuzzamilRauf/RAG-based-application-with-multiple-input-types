# Use a lightweight Python base image
FROM python:3.9-slim AS builder

# Set the working directory inside the container
WORKDIR /app/backend

# Copy and install dependencies efficiently
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY . .

# Use a lightweight runtime image to reduce final image size
FROM python:3.9-slim

# Set the working directory
WORKDIR /app/backend

# Copy only installed dependencies from the builder stage to reduce image size
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/backend /app/backend

# Expose the port your FastAPI app runs on
EXPOSE 8080

# Command to run the backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]



