import os
import cv2
from object_detector import ObjectDetector

class FaceDetector():
    def __init__(self, haar_folder = "./haar_cascade"):
        self.haar_folder = haar_folder
        #Frontal face, profile, eye and smile  haar cascade loaded
        frontal_cascade_path= os.path.join(self.haar_folder,'haarcascade_frontalface_default.xml')
        eye_cascade_path= os.path.join(self.haar_folder,'haarcascade_eye.xml')
        profile_cascade_path= os.path.join(self.haar_folder,'haarcascade_profileface.xml')
        smile_cascade_path= os.path.join(self.haar_folder,'haarcascade_smile.xml')

        #Detector object created
        # frontal face
        self.fod=ObjectDetector(frontal_cascade_path)
        # eye
        self.eod=ObjectDetector(eye_cascade_path)
        # profile face
        self.pod=ObjectDetector(profile_cascade_path)
        # smile
        self.sod=ObjectDetector(smile_cascade_path)

    def detect_objects(self, image, scale_factor, min_neighbors, min_size):
        '''
        Objects detection function
        Identify frontal face, eyes, smile and profile face and 
        display the detected objects over the image
        param: image - the image extracted from the video
        param: scale_factor - scale factor parameter for `detect` function of ObjectDetector object
        param: min_neighbors - min neighbors parameter for `detect` function 
        of ObjectDetector object
        param: min_size - minimum size parameter for f`detect` function of ObjectDetector object
        '''

        image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        eyes=self.eod.detect(image_gray,
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors,
                    min_size=(int(min_size[0]/2), int(min_size[1]/2)))

        for x, y, w, h in eyes:
            #detected eyes shown in color image
            cv2.circle(image,(int(x+w/2),int(y+h/2)),(int((w + h)/4)),(0, 0,255),3)

        # deactivated due to many false positive
        smiles=self.sod.detect(image_gray,
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors,
                    min_size=(int(min_size[0]/2), int(min_size[1]/2)))

        for x, y, w, h in smiles:
            #detected smiles shown in color image
            cv2.rectangle(image,(x,y),(x+w, y+h),(0, 0,255),3)


        profiles=self.pod.detect(image_gray,
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors,
                    min_size=min_size)

        for x, y, w, h in profiles:
            #detected profiles shown in color image
            cv2.rectangle(image,(x,y),(x+w, y+h),(255, 0,0),3)

        faces=self.fod.detect(image_gray,
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors,
                    min_size=min_size)

        for x, y, w, h in faces:
            #detected faces shown in color image
            cv2.rectangle(image,(x,y),(x+w, y+h),(0, 255,0),3)

        return image

    def extract_image_objects(self, frame):
        '''
        Extract one image from the video and then perform 
        face/eyes/smile/profile detection on the image
        param: video_file - the video from which to extract the image
        from which we extract the face
        '''
        #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return self.detect_objects(image=frame,
                scale_factor=1.3,
                min_neighbors=5,
                min_size=(50, 50))
 