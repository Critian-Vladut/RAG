#!/bin/bash

set -e  # Exit on any error

echo "Starting PDF Q&A App..."

# Remove the old containers if they exist
docker rm -f ollama 2>/dev/null || true
docker rm -f pdf-qa-app 2>/dev/null || true

# Clean up previous resources
docker-compose down --remove-orphans

# Build the app with updated code
echo "Building PDF Q&A app with updated code..."
docker-compose build pdf-qa-app

# Start Ollama container
echo "Starting Ollama container..."
docker-compose up -d ollama

echo "Waiting for Ollama to be ready..."
sleep 10

# Pull the model (ensure it's available before starting the app)
echo "Pulling LLaMA 3 model..."
docker exec ollama ollama pull llama3:8b

# Start the PDF Q&A app
echo "Starting PDF Q&A app..."
docker-compose up -d pdf-qa-app

echo "Waiting for app to start..."
sleep 10

# Done
echo "App is running at: http://localhost:8501"