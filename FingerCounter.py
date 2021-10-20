import cv2
import time
import os
import HandTrackingModule as htm
import numpy as np

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.HandDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img,flipType=True)

    fingerCheck = []
    sum1 = 0
    sum2 = 0
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1,flipType=True) #with flip the Image
        #print(fingers1)

        #calculate the sum of fingers
        sum1 = sum(fingers1)
        fingerCheck.append(fingers1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2,flipType=True) #with flip the image
            #print(fingers2)

            #calculate the sum of fingers
            sum2 = sum(fingers2)
            fingerCheck.append(fingers2)
        print(fingerCheck)

    #Calcualte the total finger of hands
    totalValue = sum1+sum2

    cv2.rectangle(img, (0, 0), (250, 200), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(totalValue), (30, 150), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    #Create the extracted image
    h,w,c = img.shape
    opImg = np.zeros([h,w,c])
    opImg.fill(255)
    if detector.results.multi_hand_landmarks:
        for handLms in detector.results.multi_hand_landmarks:
            detector.mpDraw.draw_landmarks(opImg, handLms, detector.mpHands.HAND_CONNECTIONS,
                                            detector.mpDraw.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
                                            detector.mpDraw.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=4)
                                                )

    #Calcualte the FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    #Display the Image
    cv2.imshow("Image", img)
    cv2.imshow("Extracted Image",opImg)
    if cv2.waitKey(1) == 27: 
        break
cap.release()
cv2.destroyAllWindows()
