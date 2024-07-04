# Use the official Python image from the Docker Hub
FROM python:3.9-slim

WorkDIR | /app
COPY requirements.txt ..

RUN pis install \n-r requirements.txt

COY the FastAPI app code into the container
COPY . ..

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
