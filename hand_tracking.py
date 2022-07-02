# from cvzone.HandTrackingModule import HandDetector
# import cv2
#
# cap = cv2.VideoCapture(0)
# detector = HandDetector(detectionCon=0.8, maxHands=2)
# while True:
#     # Get image frame
#     success, img = cap.read()
#     # Find the hand and its landmarks
#     hands, img = detector.findHands(img)  # with draw
#     # hands = detector.findHands(img, draw=False)  # without draw
#
#     if hands:
#         # Hand 1
#         hand1 = hands[0]
#         lmList1 = hand1["lmList"]  # List of 21 Landmark points
#         bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
#         centerPoint1 = hand1['center']  # center of the hand cx,cy
#         handType1 = hand1["type"]  # Handtype Left or Right
#
#         fingers1 = detector.fingersUp(hand1)
#
#         if len(hands) == 2:
#             # Hand 2
#             hand2 = hands[1]
#             lmList2 = hand2["lmList"]  # List of 21 Landmark points
#             bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
#             centerPoint2 = hand2['center']  # center of the hand cx,cy
#             handType2 = hand2["type"]  # Hand Type "Left" or "Right"
#
#             fingers2 = detector.fingersUp(hand2)
#
#             # Find Distance between two Landmarks. Could be same hand or different hands
#             length, info, img = detector.findDistance(
#                 lmList1[8], lmList2[8], img)  # with draw
#             # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
#     # Display
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)
# cap.release()
# cv2.destroyAllWindows()

print("hello world")
import cv2
import mediapipe as mp
import time
#
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist




def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
