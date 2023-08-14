import cv2

import os

import numpy as np

from helpers import resizeAndPad

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    
    if (len(faces) == 0):
        return None, None
    
    (x, y, w, h) = faces[0]
    
    return resizeAndPad(gray[y:y+w, x:x+h], (280,280),127), faces[0]

def prepare_training_data(data_folder_path):
    
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    #list to hold subject labels
    subjects = ["",]

    index = 0
    
    for dir_name in dirs:  
        label = dir_name
        
        subjects.append(label)
        index = index + 1
        
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        print(subject_images_names)
        
        for image_name in subject_images_names:
            
            if image_name.startswith("."):
                continue
            
            image_path = subject_dir_path + "/" + image_name
            print(image_path)

            image = cv2.imread(image_path)

            new_image = resizeAndPad(image, (400, 400), 127)
            
            # cv2.imshow("Training on image...", new_image)
            # cv2.waitKey(400)
            # cv2.destroyAllWindows()
            
            face, rect = detect_face(new_image)
            
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(index)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels, subjects


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    print(text)

def predict(test_img, face_recognizer, subjects):
    img = test_img.copy()
    face, rect = detect_face(img)

    label, confidence = face_recognizer.predict(face)
    
    label_text = subjects[label]
    print("Confindence: ", confidence )
   
    draw_rectangle(img, rect)
    
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img