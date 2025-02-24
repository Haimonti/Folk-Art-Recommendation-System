FROM python:3.9

WORKDIR /app

# Copy requirements file and install dependencies
COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Flask will run on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "app.app:app", "-b", "0.0.0.0:5000"]
