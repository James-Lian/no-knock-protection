# A computer vision script that notifies if a person is approaching the door
# good for doorbell & delivery notifs, or if parents want to enter the room without knocking lol

import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# accessing the webcam
cv2.startWindowThread()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # reading the frame

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    boxes, weights = hog.detectMultiScale(frame, winStride=(2, 2))

    vertices = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow('webcam', frame) # display
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)