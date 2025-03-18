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
│   │   └── VTD/                # Vehicle and Traffic Detection
│   │       └── yolov8n.pt      # YOLOv8 pre-trained model
│   ├── main.py
│   └── ...
├── results/
│   └── ... (generated result images and videos)
├── uploads/
│   └── ... (uploaded images)
├── venv/
├── VOI/                        # Vehicle Orientation Identification
│   ├── models/
│   │   ├── orientation_model.h5
│   │   └── orientation_model.keras
│   ├── predict_orientation.py
│   └── cmd.md
├── VTD/                        # Vehicle Detection Scripts
│   ├── cmd-vtd.md             # Command documentation
│   ├── verify_yolov8_image.py # Image detection verification
│   └── verify_yolov8_video.py # Video detection verification
└── ...

## Root
- `file_structure.md`
