import cv2 
import numpy as np
import torch


model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'C:/Users/Shivendra Pandey/Desktop/ps/projects/wildefire/best.pt', force_reload=True)



capt = cv2.VideoCapture('C:\\Users\\Shivendra Pandey\\Desktop\\ps\\projects\\wildefire\\1.mp4')
while capt.isOpened():
    ret, frame = capt.read()
    # apply our model here 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
capt.release()
cv2.destroyAllWindows()





