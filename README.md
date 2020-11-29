# THT1-RSC
Tugas 1 program untuk mendeteksi objek-objek dengan Image AI

1. Image sebelum diproses

![Gambar sebelum diproses](image1.jpg)

2. Image setelah diproses

![Gambar setelah diproses](imagenew.jpg)

3. Source Code

```python
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image1.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"])
```
