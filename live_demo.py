import tensorflow as tf 
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

background = None
accumulated_weight = 0.5
num_frames =0

ROI_top = 75
ROI_bottom = 350
ROI_right = 300
ROI_left = 590

model =tf.keras.models.load_model(r'D:\PythonWorkspace\SignLanguage\sign_language.h5')

class_labels = ['Angry','Happy','Disgust','Fear','Neutral','Sad','Surprise']

word_dict = {0:'A',1:'B',2:'C',3:'C',4:'D',5:'E',6:'F',7:'G',8:'H',9:'I',10:'J',11:'K',12:'L',13:'M',14:'N',15:'O',16:'P',17:'Q',18:'R',19:'S',20:'T',21:'U',22:'V',23:'W',24:'X',25:'Y',26:'Z',27:'Invalid'}


def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255,cv2.THRESH_BINARY)
    
     #Fetching contours in the frame (These contours can be of hand or any other object in foreground) …

    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('ASL_Alphabets.mp4')

alphabets = 'abcdefghijklmnopqrstuvwxyz'
mapping_letter = {}

for i,l in enumerate(alphabets):
    mapping_letter[l] = i
mapping_letter = {v:k for k,v in mapping_letter.items()}


while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    
    # ROI from the frame
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
    
    if num_frames < 70:
        
        cal_accum_avg(gray_frame, accumulated_weight)
        
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",
  (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
    else: 

        image = cv2.resize(gray_frame, (28, 28))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        
        pred = model.predict(image, verbose=0)
        
        for ind, i in enumerate(pred):
            print(word_dict[np.argmax(i)], end='   ')
            letter = word_dict[np.argmax(i)]

        #print(np.argmax(pred,axis=1))
        #pred_val = np.argmax(pred)
        #print(pred_val)
        #if(pred_val > 25):
        #    pred_val = 25
        #pred_letters = mapping_letter[pred_val]
        #print(pred_letters)
        
        cv2.putText(frame_copy, letter,(170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
    # Draw ROI on frame_copy
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,
    ROI_bottom), (255,128,0), 3)

    # incrementing the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.putText(frame_copy, "Hand sign recognition..",(10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    cv2.imshow("Sign Detection", frame_copy)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


























