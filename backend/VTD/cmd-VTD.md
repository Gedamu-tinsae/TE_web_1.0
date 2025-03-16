

- cd 'C:\Users\80\Desktop\sem 8\code\VTMI'
- conda create --prefix ./vtdenv python=3.10 
- conda activate ./vtdenv
- conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
- pip install ultralytics
- pip freeze > requirements.txt
- conda list --export > conda-requirements-vtd.txt

Verifying Installation:
- python verify_yolov8_image.py
- python verify_yolov8_video.py

---

Basic usage: 
    - python verify_yolov8_video.py (uses webcam)
With video file: 
    - python verify_yolov8_video.py --video path/to/video.mp4
Save output: 
    - python verify_yolov8_video.py --video path/to/video.mp4 --save
Different model: 
    - python verify_yolov8_video.py --model yolov8m.pt

- python VTD\verify_yolov8_video.py --video 'C:\Users\80\Desktop\sem 8\code\VTMI\test_images\10_sec_2.mp4' --save