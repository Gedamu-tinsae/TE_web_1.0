# Car Plate Extractor Project Structure

## Frontend
`frontend/`
├── public/
│   ├── index.html
│   └── ...
├── src/
│   ├── assets/
│   │   ├── video-icon.png
│   │   ├── file-icon.png
│   │   ├── realtime-icon.png
│   │   ├── processing-icon.png
│   │   ├── reload-icon.png
│   │   └── expand-icon.png
│   ├── components/
│   │   ├── Navbar.js
│   │   └── ...
│   ├── pages/
│   │   ├── AboutPage.js
│   │   ├── DbPage.js
│   │   ├── DocsPage.js
│   │   ├── HomePage.js
│   │   └── RealtimeDetection.js
│   ├── styles/
│   │   ├── App.css
│   │   ├── HomePage.css
│   │   ├── Navbar.css
│   │   └── index.css
│   ├── App.js
│   ├── index.js
│   └── ...
├── .gitignore
└── ...

## Backend
`backend/`
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── upload.py
│   │   │   ├── upload_video.py
│   │   │   ├── upload_tensorflow.py 
│   │   │   ├── realtime.py
│   │   │   ├── example.json
│   │   │   └── ...
│   ├── models/
│   │   ├── opencv_tensorflow.py
│   │   ├── tensorflow_model.py  
│   │   ├── plate_correction.py
│   │   ├── saved_model.pb
│   │   ├── gf.py               # Guided Filter implementation
│   │   ├── haze_removal.py     # Haze/Fog Removal implementation
│   │   ├── vehicle_type.py     # Vehicle Type Detection using YOLOv8
│   │   ├── VTD/                # Vehicle and Traffic Detection
│   │   │   └── yolov8n.pt      # YOLOv8 pre-trained model
│   │   └── VOI/                # Vehicle Orientation Identification
│   │       ├── models/
│   │       │   └── orientation_model_converted_saved_model/
│   │       │       ├── variables/
│   │       │       ├── saved_model.pb
│   │       │       └── ...
│   │       ├── orientation_model.h5
│   │       ├── orientation_model.keras
│   │       ├── orientation_model_converted_architecture.json
│   │       ├── load-about.md
│   │       └── predict_orientation.py
│   ├── main.py
│   └── ...
├── results/
│   └── ... (generated result images and videos)
├── uploads/
│   └── ... (uploaded images)
├── venv/
│   └── ...
├── VMI/                        # Vehicle Make and Model Identification
│   ├── VMI_models/             # Directory for saved models
│   │   ├── make_classes.json         # Class mapping for make classification
│   │   ├── make_mobilenet_weights.h5 # Make classification model weights
│   │   ├── final_make_mobilenet_weights.h5 # Final make classification weights
│   │   └── make_mobilenet_summary.txt    # Model architecture summary
│   ├── cmd-vmi.md              # Command line instructions for VMI
│   ├── model_files_explanation.md # Explanation of model files and formats
│   └── predict.py              # Script for prediction with trained model
├── VOI/                        # Vehicle Orientation Identification
│   ├── predict_orientation.py
│   └── cmd.md
├── VTD/                        # Vehicle Detection Scripts
│   ├── cmd-vtd.md             # Command documentation
│   ├── verify_yolov8_image.py # Image detection verification
│   └── verify_yolov8_video.py # Video detection verification
└── ...

## Root
- `file_structure.md`
