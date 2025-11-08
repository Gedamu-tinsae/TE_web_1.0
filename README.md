# Car Plate Extractor - TE_web_1.0

A comprehensive vehicle recognition system that extracts license plates, identifies vehicle characteristics, and provides real-time detection capabilities using computer vision and deep learning.

## 🚀 Features

- **License Plate Detection & Extraction**: Automatically detect and extract license plates from images and videos
- **Vehicle Characteristics Recognition**:
  - Vehicle Make Identification (VMI)
  - Vehicle Orientation Identification (VOI)
  - Vehicle Type Detection (VTD) using YOLOv8
  - Vehicle Color Detection
- **Multiple Processing Methods**:
  - OpenCV-based detection
  - TensorFlow-based detection
- **Real-time Detection**: Live camera feed processing for real-time vehicle and plate detection
- **Low Visibility Enhancement**: Special processing for images/videos captured in fog, haze, or poor lighting conditions
- **Video Processing**: Process entire video files for batch analysis
- **User-Friendly Web Interface**: Modern React-based frontend with intuitive controls

## 🛠️ Tech Stack

### Frontend
- **React** 18.2.0
- **React Router** for navigation
- **Axios** for API communication
- **CSS3** for styling

### Backend
- **FastAPI** - High-performance Python web framework
- **TensorFlow** & **Keras** - Deep learning models
- **PyTorch** & **YOLOv8** - Object detection
- **OpenCV** - Computer vision operations
- **EasyOCR** - Optical character recognition
- **NumPy**, **Pandas**, **SciPy** - Data processing
- **Uvicorn** - ASGI server

## 📋 Prerequisites

- **Python** 3.8 or higher
- **Node.js** 14.x or higher
- **npm** or **yarn**
- **Git**

## 🚀 Installation & Setup

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Gedamu-tinsae/TE_web_1.0.git
   cd TE_web_1.0/backend
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the backend server**
   ```bash
   uvicorn app.main:app --reload
   ```
   
   The backend API will be available at `http://127.0.0.1:8000`

### Frontend Setup

1. **Navigate to the frontend directory**
   ```bash
   cd ../frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm start
   ```
   
   The frontend will be available at `http://localhost:3000`

## 📖 Usage

1. **Launch both backend and frontend servers** following the setup instructions above

2. **Choose your detection method**:
   - **Image Upload**: Upload a single image for plate detection
   - **Video Upload**: Upload a video file for batch processing
   - **Real-time Detection**: Use your webcam for live detection

3. **Select processing options**:
   - Choose between OpenCV or TensorFlow processing methods
   - Enable low visibility enhancement for poor lighting conditions

4. **View results**:
   - See the original and annotated media side-by-side
   - View extracted license plate information
   - Check vehicle characteristics (make, type, color, orientation)
   - Review confidence scores for each detection

## 📁 Project Structure

```
TE_web_1.0/
├── backend/
│   ├── app/
│   │   ├── api/endpoints/      # API route handlers
│   │   ├── models/             # ML models and processing logic
│   │   └── main.py             # FastAPI application
│   ├── VMI/                    # Vehicle Make Identification
│   ├── VOI/                    # Vehicle Orientation Identification
│   ├── VTD/                    # Vehicle Type Detection
│   ├── requirements.txt        # Python dependencies
│   └── README.md              # Backend documentation
├── frontend/
│   ├── public/                # Static assets
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/             # Page components
│   │   ├── styles/            # CSS stylesheets
│   │   └── App.js             # Main React component
│   └── package.json           # Node.js dependencies
└── file_structure.md          # Detailed project structure
```

## 🔌 API Endpoints

### Main Endpoints

- `POST /api/upload` - Upload image for license plate extraction
- `POST /api/upload-video` - Upload video for processing
- `POST /api/upload-tensorflow` - Upload media using TensorFlow processing
- `POST /api/realtime` - Real-time detection endpoint
- `GET /results/{filename}` - Retrieve processed results

For detailed API documentation, visit `http://127.0.0.1:8000/docs` when the backend is running.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 👥 Authors

- Gedamu Tinsae - [GitHub Profile](https://github.com/Gedamu-tinsae)

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for vehicle detection
- TensorFlow and Keras for deep learning models
- FastAPI for the excellent web framework
- React community for frontend resources

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the repository owner.

---

**Note**: Make sure to configure the proxy in `frontend/package.json` to match your backend server address if running on a different host/port.
