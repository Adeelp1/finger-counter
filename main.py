import cv2 as cv
import mediapipe as mp
import pygame

# Initialize pygame mixer
pygame.mixer.init()

mpHands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

# Initialize Hands object
hands = mpHands.Hands(
        static_image_mode= False,
        max_num_hands= 1,
        min_detection_confidence= 0.7
    )

def getHandlandMarks(img, drw):
    lmlist = []
    frameRgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    handsDetected = hands.process(frameRgb)

    if handsDetected.multi_hand_landmarks:
        for lanmarks in handsDetected.multi_hand_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(lanmarks.landmark):
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append((id, cx, cy))
        if drw:
            drawing.draw_landmarks(
                img,
                lanmarks,
                mpHands.HAND_CONNECTIONS
            )        
    return lmlist

# Finger Counting

def fingerCounting(lmlist):
    if len(lmlist) < 21: # Ensure all landmarks are detected
        return None

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

# play sound
def voice(num):
    sound_files = {
        0: 'sound/0.mp3',
        1: 'sound/1.mp3',
        2: 'sound/2.mp3',
        3: 'sound/3.mp3',
        4: 'sound/4.mp3',
        5: 'sound/5.mp3'
    }

    file_path = sound_files.get(num)
    if file_path:
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
        except pygame.error as e:
            print(f"Error loading sound: {e}")

#camera setup
cam = cv.VideoCapture(0)
last_count = -1 # Track the last count to avoid repeating sounds


while True:
    success, frame = cam.read()

    if not success:
        print("cam not detected...!")
        continue

    frame = cv.flip(frame, 1)
    lmlist = getHandlandMarks(img=frame,drw=False)

    if lmlist:
        fc = fingerCounting(lmlist)
        # Play sound only if the count changes
        if fc is not None:
            if fc != last_count:
                voice(fc)
                last_count = fc # Update last count

        # Display count on frame
        cv.rectangle(frame, (400,10), (600,250), (0,0,0), -1)
        cv.putText(frame, str(fc), (400,250), cv.FONT_HERSHEY_PLAIN, 20, (0,255,255), 30)

    cv.imshow("AI finger counter", frame)
    if cv.waitKey(1) == ord('q'):
        break

hands.close()
cam.release()
cv.destroyAllWindows()
