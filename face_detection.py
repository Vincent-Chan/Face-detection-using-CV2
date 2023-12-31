#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary libraries/packages

import numpy as np
import cv2


# In[2]:


def main():
    
    # Pretrained data by cv2 can be found in the following website
    # https://github.com/opencv/opencv/tree/4.x/data/haarcascades
    
    # Load pre-trained face data
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Load pre-trained smile data
    smile = cv2.CascadeClassifier('haarcascade_smile.xml')

    # Load pre-trained left eye data
    left_eye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

    # Load pre-trained right eye data
    right_eye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    
    
    # To capture video from webcam
    webcam = cv2.VideoCapture(0)
    
    
    # BGR value for different colors
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)
    turquoise = (208, 224, 64)
    black = (0, 0, 0)
    
    
    print("Press 'q' key if you want to terminate the program")
    
    
    # Iterate for infinitely many times
    while True:
        
        # Get the current frame from the webcam video stream
        successful_frame_read, frame = webcam.read()
        
        # in case there are any errors, we terminate break the while loop
        if not successful_frame_read:
            print('Some errors occured!')
            break
            
        # convert the image to a grayscale so as to make the calculation more efficient
        grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # recognise faces
        # https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
        faces = face.detectMultiScale(grayscaled_image)
        
        
        # run face recognition for each face
        for (x, y, w, h) in faces:
            
            # draw a red rectangle around the face
            # https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
            cv2.rectangle(frame, (x, y), (x+w, y+h), red, 4)
            
            # get the sub-frame (by using numpy N-dimensional array slicing)
            face_slice = frame[y:y+h, x:x+h]
            
            # change the face_slice to grayscale as well
            face_slice_grayscale = cv2.cvtColor(face_slice, cv2.COLOR_BGR2GRAY)
            
            
            smiles = smile.detectMultiScale(face_slice_grayscale, scaleFactor = 1.7, minNeighbors = 20)
            
            left_eyes = left_eye.detectMultiScale(face_slice_grayscale, scaleFactor = 1.7, minNeighbors = 20)
            
            right_eyes = right_eye.detectMultiScale(face_slice_grayscale, scaleFactor = 1.7, minNeighbors = 20)
            
            
            # find all smiles in the face
            for (x_, y_, w_, h_) in smiles:
                
                # draw rectangles around those smiles
                cv2.rectangle(face_slice, (x_, y_), (x_ + w_, y_ + h_), green, 3)
                
                # label the face that are smiling using turquoise text
                if len(smiles) > 0:
                    cv2.putText(frame, "Smiling", (x, y+h+50), fontScale = 2, fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = turquoise, thickness = 5)
            
            # find all left eyes in the face
            for (x_, y_, w_, h_) in left_eyes:
                
                # draw yellow rectangles around those left eyes
                cv2.rectangle(face_slice, (x_, y_), (x_ + w_, y_ + h_), yellow, 2)
            
            # find all right eyes in the face
            for (x_, y_, w_, h_) in right_eyes:
                
                # draw blue rectangles around those right eyes
                cv2.rectangle(face_slice, (x_, y_), (x_ + w_, y_ + h_), blue, 2)
                
    
        
        # show the image
        # https://www.geeksforgeeks.org/python-opencv-cv2-imshow-method/
        cv2.imshow("Face detection", frame)
            
        # From GeeksForGeeks: 
        # waitkey() function of Python OpenCV allows users to display a window for given milliseconds or until any key is pressed
        # https://www.geeksforgeeks.org/python-opencv-waitkey-function/
        key = cv2.waitKey(1)
            
        # Terminate the program if the 'q' key is pressed
        if key == ord('q') or key == ord('Q'):
            break

    
    webcam.release()
    cv2.destroyAllWindows()
    
    print("\nThanks for using face detection application!")


# In[3]:


if __name__ == "__main__":
    main()

