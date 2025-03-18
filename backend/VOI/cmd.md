- python "c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\process_orientation.py" --quick
- python "c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\process_orientation.py" --max_per_category 200
- python "c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\process_orientation.py" --quick --extract_samples

---
- python 'c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\train_orientation_model.py' --epochs 30 --batch_size 32 --learning_rate 0.001

# i used(because gpu limit): first trained with this then with the above
 -  python 'c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\train_orientation_model.py' --batch_size 8 --image_size 128 --model_size tiny --epochs 30    

- python 'c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\evaluate_orientation_model.py'

- # For a single image
python 'c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\predict_orientation.py' --input path/to/your/image.jpg --output path/to/save/results

# For a directory of images
python 'c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\predict_orientation.py' --input path/to/your/image/directory --output path/to/save/results

---
- python 'c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\check_gpu.py'

- python 'c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\predict_orientation.py' --input "C:\Users\80\Desktop\sem 8\code\VOI\test images\Cars416.png" --output "C:\Users\80\Desktop\sem 8\code\VOI\test images\results"

# Using the simplified test script
python 'c:\Users\80\Desktop\sem 8\code\VOI\test_image_orientation.py' --image "C:\Users\80\Desktop\sem 8\code\VOI\test images\Cars416.png"

---

 python 'c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\predict_orientation.py' --input "C:\Users\80\Desktop\sem 8\code\VOI\test images\image1.jpg" --output "C:\Users\80\Desktop\sem 8\code\VOI\test images\results"


----
(C:\Users\80\Desktop\sem 8\code\VOI\env) PS C:\Users\80\Desktop\sem 8\code\VOI> 
----
python 'c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\predict_orientation.py' --input "C:\Users\80\Desktop\sem 8\code\VOI\test images\image5.jpg" --output "C:\Users\80\Desktop\sem 8\code\VOI\test images\results"

python 'c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\scripts\predict_orientation.py' --input "C:\Users\80\Desktop\sem 8\code\VOI\test images\image6.png" --output "C:\Users\80\Desktop\sem 8\code\VOI\test images\results" 