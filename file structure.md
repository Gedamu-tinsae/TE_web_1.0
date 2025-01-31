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
│   │   ├── drive-icon.png
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
│   ├── app/
│   │   ├── api/
│   │   │   ├── endpoints/
│   │   │   │   ├── upload.py
│   │   │   │   ├── example.json
│   │   │   │   └── ...
│   │   ├── models/
│   │   │   ├── opencv_tensorflow.py
│   │   │   └── ...
│   │   ├── main.py
│   │   └── ...
│   ├── results/
│   │   └── ... (generated result images)
│   ├── uploads/
│   │   └── ... (uploaded images)
│   └── ...
├── .gitignore
└── ...

## Root
- `file structure.md`

Note: This structure will be updated as new files and features are added to the project.
