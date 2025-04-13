# Use a base image like Python
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the local directory to /app in the container
COPY . /app

# Install any dependencies (if you have requirements.txt)
RUN pip install -r requirements.txt

# Command to run your application
CMD ["python", "your_script.py"]
