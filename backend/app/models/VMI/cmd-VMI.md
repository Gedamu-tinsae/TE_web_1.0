- cd 'C:\Users\80\Desktop\sem 8\code\VTMI\'
- conda create --prefix ./VMIenv python=3.10
- conda activate ./vmienv
- conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
- python -m pip install 'tensorflow<2.11'
- pip install "numpy<2"
- pip install opencv-python
- pip install matplotlib
- pip install pandas
- pip install scikit-learn
- pip install seaborn
- conda list --export > conda-requirements-vmi.txt

---

format the coco json file
- Shift + Alt + F 

--- 

Set QUICK_TEST = False: for full training
Set QUICK_TEST = True: for full quick training

---
- python vmi/explore_dataset.py
- python VMI/train_model.py
- python vmi/predict.py
- python VMI/troubleshoot_model.py --all

--
- python VMI/predict.py --image test_images/your_car.jpg --debug
- python VMI/predict.py --image test_images/your_car.jpg --base-model mobilenet

python VMI/predict.py --image test_images/car2.jpg --debug
python VMI/predict.py --image test_images/car3.png --base-model mobilenet
python VMI/predict.py --image test_images/val_img.jpg --base-model mobilenet


(C:\Users\80\Desktop\sem 8\code\VTMI\vmienv) PS C:\Users\80\Desktop\sem 8\code\VTMI> python VMI/predict.py --image test_images/t/bm.jpg --base-model mobilenet

---

# Analyze prediction errors
python VMI/predict_analyze.py --image test_images/car2.jpg --true-class BMW --base-model mobilenet
python VMI/predict_analyze.py --image test_images/car3.png --true-class Mitsubishi --base-model mobilenet

---

# Try Different Model Architectures
python VMI/predict.py --image test_images/car2.jpg --base-model efficientnet
python VMI/predict.py --image test_images/car3.png --base-model efficientnet
python VMI/predict.py --image test_images/car2.jpg --base-model resnet

# Retrain with different parameters (after editing train_model.py)
# Increase regularization and class weighting
python VMI/train_model.py --base-model efficientnet

# Enhanced data augmentation (first modify augmentation settings)
# Edit train_model.py to increase data augmentation parameters
python VMI/train_model.py --strong-augmentation

# Check class distribution to identify imbalances
python VMI/troubleshoot_model.py --analyze-data

# Enhanced model inspection
python VMI/predict.py --image test_images/car2.jpg --deep-analyze --base-model efficientnet