
# My ML Project

## Setup

1. Clone the repository
2. Install dependencies
   
   pip install -r requirements.txt
   
3. Run the model script
   
   python src/model.py

## Docker

To build and run the docker container:
/
 1. Build the Docker image
   
   docker build -t my_ml_project .
   
2. Run the Docker container 
   
   docker run -p 4000:80 my_ml_project
   
