# Use the provided base image
FROM svizor/zoomcamp-model:3.11.5-slim

# Set the working directory
WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock to the working directory
COPY Pipfile Pipfile.lock ./

# Install dependencies using pipenv
RUN pipenv install --system --deploy

# Copy your Flask application script to the working directory
COPY app_2.py ./

# Expose the port that the Flask app will run on
EXPOSE 9696

# Define the default command to run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "app_2:app"]





