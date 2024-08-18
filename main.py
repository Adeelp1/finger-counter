import cv2 as cv
import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as drawing

#Get Handlandmarks
def getHandlandMarks(img,drw):
    lmlist = []

    hands = mpHands.Hands(
        static_image_mode= False,
        max_num_hands= 1,
        min_detection_confidence= 0.7
    )

    frameRgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    handsDetected = hands.process(frameRgb)
    if handsDetected.multi_hand_landmarks:
        for lanmarks in handsDetected.multi_hand_landmarks:
            # print(lanmarks)
            for id, lm in enumerate(lanmarks.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append((id, cx, cy))
                # print(lmlist)
        if drw:
            drawing.draw_landmarks(
                img,
                lanmarks,
                mpHands.HAND_CONNECTIONS
            )        
    return(lmlist)

# Finger Counting

def fingerCounting(lmlist):
    count = 0
    if lmlist[8][2] < lmlist[6][2]:
        count += 1
    if lmlist[12][2] < lmlist[10][2]:
        count += 1
    if lmlist[16][2] < lmlist[14][2]:
        count += 1
    if lmlist[20][2] < lmlist[18][2]:
        count += 1
    if lmlist[4][1] < lmlist[2][1]:
        count += 1

    return count



#camera setup

cam = cv.VideoCapture(0)


while True:
    success, frame = cam.read()

    if not success:
        print("cam not detected...!")
        continue
    frame = cv.flip(frame, 1)
    lmlist = getHandlandMarks(img=frame,drw=False)

    if lmlist:
        # print(lmlist)
        fc = fingerCounting(lmlist=lmlist)
        # print(fc)
        cv.rectangle(frame, (400,10), (600,250), (0,0,0), -1)
        cv.putText(frame, str(fc), (400,250), cv.FONT_HERSHEY_PLAIN, 20, (0,255,255), 30)

    cv.imshow("AI finger counter", frame)
    if cv.waitKey(1) == ord('q'):
        break
cam.release()
cv.destroyAllWindows()
