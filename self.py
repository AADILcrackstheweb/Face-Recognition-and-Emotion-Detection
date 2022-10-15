import face_recognition
from fer import FER
import matplotlib.pyplot as plt 
from cv2 import (VideoCapture, namedWindow, imshow, waitKey, destroyWindow, imwrite)
# initialize the camera
# If you have multiple camera connected with 
# current device, assign a value in cam_port 
# variable according to that
cam_port = 0
cam = VideoCapture(cam_port)
  
# reading the input using the camera
result, image = cam.read()
6  
# If image will detected without any error, 
# show result
if result:
  
    '''# showing result, it take frame name and image 
    # output
    imshow("Your_image", image)
    print("Press 0 to safely exit and compare.")'''
  
    # saving image in local storage
    imwrite("imageface.jpg", image)
  
    ''' # If keyboard interrupt occurs, destroy image 
    # window
    waitKey(0)
    destroyWindow("Your_image") ''' 
# If captured image is corrupted, moving to else part
else:
    print("No image detected. Please! try again")
known_image = face_recognition.load_image_file("i1.jpg")
unknown_image = face_recognition.load_image_file("imageface.jpg")
biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
print(results)
test_image_one = plt.imread("imageface.jpg")
emo_detector = FER(mtcnn=True)
# Capture all the emotions on the image
captured_emotions = emo_detector.detect_emotions(test_image_one)
# Print all captured emotions with the image
print(captured_emotions)
#plt.imshow(test_image_one)
#Use the top Emotion() function to call for the dominant emotion in the image
dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
print(dominant_emotion, emotion_score)


