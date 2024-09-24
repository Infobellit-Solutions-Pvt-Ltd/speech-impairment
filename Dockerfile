# Use an official Python runtime as a parent image
FROM python:3.9
 
# Set the working directory in the container
WORKDIR /app
 
# Copy the current directory contents into the container at /app
COPY . /app
 
# Install dependencies from requirements.txt (make sure the file exists in your project)
RUN pip install --no-cache-dir -r requirements.txt
 
# Expose port 5000 to the outside world
EXPOSE 5000
 
# Define environment variables
ENV FLASK_APP=main.py
ENV FLASK_ENV=production
 
# Command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
