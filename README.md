# Age-Gender-D-VT
D-VT Age & Gender Detection App

A Python Flask API for performing real-time or image-based age and gender detection using the D-VT model.
Authors

    Bilal Alibashi Mohamed
    Mohamed Abdinur adam
    

Introduction

This project is a Python Flask API that acts as a backend for age and gender detection. It is designed to receive images or video data, process them using the D-VT model, and return the predicted age, gender, and the bounding box of any detected faces.
Features

    API Endpoints: Provides a RESTful API to handle image and video data.

    D-VT Integration: Utilizes the D-VT age and gender detection model on the server-side.

    Flexible Input: Can accept image files directly or be configured for video stream processing.

    JSON Output: Returns results in a structured JSON format, including predicted age, gender, and confidence scores.

Technical Stack
Backend (API)

    Python: The primary programming language for the server.

    Flask: A lightweight web framework for building the REST API.

    D-VT: The age and gender estimation model.

    PyTorch: The machine learning framework used to load and run the D-VT model.

    Other Dependencies: Pillow, Numpy, opencv-python.

Prerequisites

To run this project, you will need to have the following installed:

    Python 3.8 or higher

    A code editor (e.g., VS Code)

Setup Instructions

Follow these steps to get a copy of the project up and running on your local machine.

    Clone the repository:

    git clone https://github.com/your-username/dvt_project.git
    cd dvt_project/backend

    Create and activate a virtual environment:

    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS / Linux
    source venv/bin/activate

    Install the required Python packages:

    pip install -r requirements.txt

    Run the Flask application:

    python demo.py


Contributing

We welcome contributions! Please feel free to open issues or submit pull requests.
License

This project is licensed under the MIT License - see the LICENSE file for details.
