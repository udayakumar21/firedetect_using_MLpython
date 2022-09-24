from cgitb import grey
import cv2
from playsound import playsound
fire_cascade = cv2.CascadeClassifier("fire_detection.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frames = cap.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(frames, 1.03, 6)

    for x, y, w, h in fire:
        cv2.rectangle(frames, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frames[x:x+w, y:y+h]
        print("Fire detected!!")
        playsound("audio.mp3")
    cv2.imshow("fire detected", frames)
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()
