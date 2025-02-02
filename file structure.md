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
│   │   └── HomePage.js
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
│   │   │   ├── example.json
│   │   │   └── ...
│   ├── models/
│   │   ├── opencv_tensorflow.py
│   │   ├── tensorflow_model.py  
│   │   └── saved_model.pb
│   ├── main.py
│   └── ...
├── results/
│   └── ... (generated result images and videos)
├── uploads/
│   └── ... (uploaded images)
└── ...

## Root
- `file structure.md`

Note: This structure will be updated as new files and features are added to the project.
