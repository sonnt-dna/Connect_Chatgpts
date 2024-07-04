For Build Docker file:
Step 1: docker build -t my_image .
Step 2: docker run -p 8017:3500 my_image uvicorn app:app --reload --host 0.0.0.0 --port 3500  
Step 3: Test with file "Test_API_only_Test_API.ipynb" #Change your patch
