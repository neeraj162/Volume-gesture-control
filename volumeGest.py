import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

####
wCam = 640
hCam = 480
###

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]



pTime = 0
avgfps = 0
n = 0
detector = htm.handDetection(det_confid=0.7, hands=1)


vol = 0
volBar = 400
volPer = 100
area = 0
colorvol = (255,0,0)
while cap.isOpened():
    success, img = cap.read()
    img = detector.findHands(img)

    lmlist,bbx = detector.findPos(img)

    if len(lmlist)!=0:

        # FILTER BASED ON SIZE

        area = abs(bbx[2]-bbx[0]) * abs(bbx[3]-bbx[1])//100
        # print(area)
        # area-100 to 500

        if 100<area<500:
            # print("yes")

            # FIND DISTANCE BETWEEN INDEX AND THUMB
            length, img, info = detector.fingerDistance(4,8,img)


            # CONVERT VOLUME

            # print(length)
            # length = 15 - 130
            # VOLRANGE = -63.5 - 0

            # vol = np.interp(length,[15,130],[minVol, maxVol])
            volBar = np.interp(length, [15,125], [400,150])
            volPer = np.interp(length, [15,125], [0,100])

            # REDUCING RESOLUTION TO MAKING SMOOTH
            smooth = 5
            volPer = smooth * round(volPer/smooth)


            # CHECK FINGERS UP
            fingers = detector.fingersUp()
            # print(fingers)

            # IF LITTLE FINGER IS DOWN SET VOLUME
            if not fingers[4]:
                # volume.SetMasterVolumeLevel(vol, None)
                volume.SetMasterVolumeLevelScalar(volPer/100,None)
                cv2.circle(img, (info[4],info[5]),10, (0,255,0), cv2.FILLED)
                colorvol=(0,255,0)
            else:
                colorvol=(255,0,0)



    # DRAWINGS
    cv2.rectangle(img, (50,150), (85,400), (255,0,0),3)
    cv2.rectangle(img, (50,int(volBar)), (85, 400), (255,0,0),cv2.FILLED)
    cv2.putText(img, str(f"{int(volPer)}%"), (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 1, 255), 3)
    cvol = int(volume.GetMasterVolumeLevelScalar()*100)
    cv2.putText(img, str(f"Vol Set:{cvol}%"), (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, colorvol, 3)


    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    avgfps = avgfps + fps
    n = n + 1

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 1, 255), 3)

    cv2.imshow("Image something", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
print(f"Average fps: {avgfps / n}")
cv2.destroyAllWindows()