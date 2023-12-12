import cv2
from ffpyplayer.player import MediaPlayer
import time

# previous result
audio_prediction_labels = ['sad', 'sad', 'calm', 'calm', 'disgust', 'calm', 'disgust', 'calm', 'calm', 'sad', 'disgust', 'disgust', 'calm', 'disgust', 'angry', 'sad', 'calm', 'disgust', 'calm', 'disgust', 'disgust', 'disgust', 'sad', 'calm', 'calm', 'calm', 'disgust', 'calm', 'calm', 'calm', 'happy', 'happy', 'happy']
visual_prediction_labels = ['surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'disgust', 'disgust', 'disgust', 'fear', 'surprise', 'surprise', 'surprise', 'surprise', 'sad', 'happy', 'happy', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'sad', 'surprise', 'surprise', 'surprise', 'surprise']
combined_labels = ['surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'disgust', 'disgust', 'disgust', 'fear', 'surprise', 'surprise', 'surprise', 'surprise', 'sad', 'happy', 'happy', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'sad', 'surprise', 'surprise', 'surprise', 'surprise']

cap = cv2.VideoCapture("demo.mp4")
player = MediaPlayer("demo.mp4")

frame_count = 0
while(True):
    frame_count += 1
    # Capture frames in the video
    ret, frame = cap.read()
    audio_frame, val = player.get_frame()
    
    time.sleep(1/80)

    # describe the type of font
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX

    interval_count = min(frame_count // 45, 32)
    
    visual_text = "Visual: " + visual_prediction_labels[interval_count]
    audio_text = "Audio: " + audio_prediction_labels[interval_count]
    combined_text = "Combined: " + combined_labels[interval_count]

    frame = cv2.flip(frame, 0)

    # Use putText() method for
    # inserting text on video
    cv2.putText(frame, 
                visual_text, 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
    
    cv2.putText(frame, 
                audio_text, 
                (50, 100), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
    
    cv2.putText(frame, 
                combined_text, 
                (50, 150), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)

    # Display the resulting frame
    cv2.imshow('video', frame)
  
    # creating 'q' as the quit button for the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # display audio
    if val != 'eof' and audio_frame is not None:
        img, t = audio_frame
# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()