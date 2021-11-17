# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 23:38:51 2021

@author: Santosh
"""

import cv2 as cv
import os

import face_recognition

# install CMake
# install face_recognition

image_path = "C:/MEDIA_PIPE/FACE_RECOGNITION"


images = []
lables = []
for file in os.listdir(image_path):
    print(file)
    full_path = os.path.join(image_path, file)
    
    image = cv.imread(full_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    images.append(image)
    lables.append(file)
    
    
print(lables)  


face_encodings = []
for image in images:
        face_encoding = face_recognition.face_encodings(image)[0]
        face_encodings.append(face_encoding)
        
        
video_path =  "C:/MEDIA_PIPE/Video/video1.mp4"     

capture= cv.VideoCapture(video_path)

scale = 0.25
face_score = 0.6

while True:
    result, image_read = capture.read()
    if result:
        
        image_rgb = cv.cvtColor(image_read, cv.COLOR_BGR2RGB)
        
        image_rgb = cv.resize(image_rgb, (0,0), None, scale, scale)
        
        try:         
            faceLoc = face_recognition.face_locations(image_rgb)
            faceEnc = face_recognition.face_encodings(image_rgb, faceLoc)
            
            for face_l, face_e  in zip(faceLoc, faceEnc):
                
                face_match = face_recognition.compare_faces(face_encodings, face_e)
                
                face_distance = face_recognition.face_distance(face_encodings, face_e)
                mav_value = max(face_match)
                max_index = face_match.index(mav_value)
                
                y1, x2, y2, x1 = face_l
                y1, x2, y2, x1 = int(y1/scale), int(x2/scale), int(y2/scale), int(x1/scale)
                
                if face_distance[max_index] < face_score:                    
                    detected_person = lables[max_index]
                    detected_person = detected_person.replace('.jpg', '')
                else:
                    detected_person = "Unknown"
                
                cv.rectangle(image_read, (x1, y1), (x2, y2), (50, 200, 50), 2)
                cv.rectangle(image_read, (x1, y1-25), (x2, y1-4), (50, 200, 50), cv.FILLED)
                cv.putText(image_read, detected_person, (x1, y1-4), cv.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        except IndexError as e:
            print('No Face Detected', e)
            
        cv.imshow("Window", image_read)
        if cv.waitKey(1) & 255 == 27:
            break
    
capture.release()
cv.destroyAllWindows()    
        

  