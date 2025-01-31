# Car Plate Extractor Backend

## Overview
The Car Plate Extractor backend is built using FastAPI and is designed to handle image uploads for the purpose of extracting license plates from car images. This document provides instructions on how to set up and run the backend application.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/car-plate-extractor.git
   cd car-plate-extractor/backend
   ```

2. **Create a Virtual Environment**
   It is recommended to create a virtual environment to manage dependencies.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   Install the required packages listed in `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
To start the FastAPI server, run the following command:
```bash
uvicorn app.main:app --reload
```
The server will start on `http://127.0.0.1:8000/`.

## API Endpoints
- **POST /upload**
  - Description: Upload an image for license plate extraction.
  - Request: Form-data with an image file.
  - Response: JSON containing the extracted license plate information.

## Usage
Once the server is running, you can use the frontend application to upload images and receive extracted license plate data.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.