# Start with a base image that has Python 3.11
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GROQ_API_KEY=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 \
    LANGCHAIN_TRACING_V2=true \
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com \
    LANGCHAIN_API_KEY=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 \
    LANGCHAIN_PROJECT=pr-sparkling-poisoning-55

# Set a working directory in the container
WORKDIR /app

# Copy your local requirements.txt file into the container
COPY requirements.txt /app/

# Install pip and dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Expose the port the app will run on (default Flask port)
EXPOSE 8080

# Set the command to run your Flask app
CMD ["python", "server.py"]