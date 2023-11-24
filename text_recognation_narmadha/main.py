
import cv2 
import easyocr 
import matplotlib.pyplot as plt 
import numpy as np

path = 'C:/Users/ASUS/Downloads/text_recognation_narmadha/img.jpg'
img = cv2.imread(path)

reader = easyocr.Reader(['en'], gpu=False)


text_ = reader.readtext(img)

threshold = 0.5#confidence value

for t_, t in enumerate(text_):
    print(t)

    bbox, text, score = t

    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0,0, 255), 2)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (100, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()