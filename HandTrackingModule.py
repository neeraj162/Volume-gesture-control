import cv2
import mediapipe as mp
import time
import math


class handDetection:
    def __init__(self, mode=False, hands=2, comp=1, det_confid=0.5, track_confid=0.5):
        self.mode = mode
        self.hands = hands
        self.complexity = comp
        self.det_confid = det_confid
        self.track_confid = track_confid

        self.MpHands = mp.solutions.hands
        self.Hands = self.MpHands.Hands(self.mode, self.hands, self.complexity, self.det_confid, self.track_confid)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipId = [4,8,12,16,20]

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.Hands.process(imgRGB)

        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handlm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlm, self.MpHands.HAND_CONNECTIONS)

        return img

    def findPos(self, img, handNum=0, draw=True):
        xlist = []
        ylist = []
        bbx=[]
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xlist.append(cx)
                ylist.append(cy)
                self.lmList.append([id, cx, cy])

            xmin,xmax = min(xlist),max(xlist)
            ymin,ymax = min(ylist),max(ylist)
            bbx = xmin,ymin,xmax,ymax
            if draw:
                cv2.rectangle(img, (bbx[0]-20,bbx[1]-20),(bbx[2]+20,bbx[3]+20), (0,255,0),2)
        return self.lmList, bbx

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipId[0]][1] > self.lmList[self.tipId[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipId[id]][2] < self.lmList[self.tipId[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # totalFingers = fingers.count(1)

        return fingers

    def fingerDistance(self, p1, p2, img, draw=True, r=10, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)
            l = math.hypot(x2 - x1, y2 - y1)

        return l, img, [x1, y1, x2, y2, cx, cy]


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    avgfps = 0
    n = 0
    detector = handDetection(det_confid=0.8)

    while cap.isOpened():
        success, img = cap.read()

        img = detector.findHands(img)

        lmlist = detector.findPos(img)

        # if len(lmlist) != 0:
            # print(lmlist)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        avgfps = avgfps + fps
        n = n + 1

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 1, 255), 2)

        cv2.imshow("Image something", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    print(f"Average fps: {avgfps / n}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
