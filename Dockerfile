# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --default-timeout=300 -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application using uvicorn
CMD python app/main.py


