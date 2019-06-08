# Official Python runtime as an image
FROM python:3.6

# EXPOSE ports on which a container will listen for connections
# Flask apps listen to port 5000  by default
# EXPOSE 5000

# Sets the working directory for following COPY and CMD instructions
# Creates a directory with this name if it doesn’t exist
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
# COPY . ./


CMD ["python", "src/app.py"]