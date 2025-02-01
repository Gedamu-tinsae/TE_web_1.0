from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import upload, upload_video
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(upload_video.router, prefix="/api")
app.mount("/results", StaticFiles(directory="results"), name="results")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Plate Extractor API"}
