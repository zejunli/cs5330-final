"""
    Team member: Luofei Shi and Zejun Li
    File: Game.py
    Description: this is the game part of the final project. A user can put a finger on the
                 screen and press s to start the game. The user need to hold the 2nd finger within
                 a limited area from the line to avoid failure. The game will end once the 
                 player go through the whole line or failed to keep the 2nd finger on the line.
"""
import pickle
import pygame
import cv2
import math
import numpy as np
import mediapipe as mp


"""
    Using the mediapipe library to find hands
"""
class HandDetector:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []


    """
        Find the hand in a BRG image.
        set fliptype to False if the input image is already flipped
    """
    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return allHands, img
        else:
            return allHands


def makeOffsetPoly(path, offset, outer_ccw=1):
    def normalizeVec(x, y):
        distance = np.sqrt(x * x + y * y)
        return x / distance, y / distance

    num_points = len(path)
    newPath = []
    for curr in range(num_points):
        prev = (curr + num_points - 1) % num_points
        next = (curr + 1) % num_points

        vnX = path[next][0] - path[curr][0]
        vnY = path[next][1] - path[curr][1]
        vnnX, vnnY = normalizeVec(vnX, vnY)
        nnnX = vnnY
        nnnY = -vnnX

        vpX = path[curr][0] - path[prev][0]
        vpY = path[curr][1] - path[prev][1]
        vpnX, vpnY = normalizeVec(vpX, vpY)
        npnX = vpnY * outer_ccw
        npnY = -vpnX * outer_ccw

        bisX = (nnnX + npnX) * outer_ccw
        bisY = (nnnY + npnY) * outer_ccw

        bisnX, bisnY = normalizeVec(bisX, bisY)
        bislen = offset / np.sqrt(1 + nnnX * npnX + nnnY * npnY)

        newPath.append([int(path[curr][0] + bislen * bisnX), int(path[curr][1] + bislen * bisnY)])

    return newPath


def isPointInLine(line, point):
    x1, y1 = line[0]
    x2, y2 = line[1]
    x, y = point
    if x1 == x2:
        if y1 > y2:
            if y2 < y < y1:
                return True
            else:
                return False
        else:
            if y1 < y < y2:
                return True
            else:
                return False
    else:
        m = (y2 - y1) / (x2 - x1)
        c = y2 - m * x2
        yNew1 = int(m * x + c)
        yNew2 = int(m * x + c)
        if yNew1 - 40 < y < yNew2 + 40:
            return True
        else:
            return False


def Game():
    # Initialize
    pygame.init()
    pygame.event.clear()

    # Create Window/Display
    width, height = 1280, 720   # same size as my macbook webcam
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Team Luofei Shi And Zejun Li")

    # Initialize Clock for FPS
    fps = 30
    clock = pygame.time.Clock()

    # Init the webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # width
    cap.set(4, 720)  # height

    # Load images from 'imgs' folder
    imgCookie = cv2.imread("./imgs/target.png")
    imgCookieCrack = cv2.imread("./imgs/crack.png")
    imgGameOver = cv2.imread("./imgs/failed.png")
    imgGameWon = cv2.imread("./imgs/pass.png")

    # Setup hand Detector
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    # Flags / Stats
    gameStart, gameOver, gameWon = False, False, False

    # Parameter
    difficulty = 20  # lower is harder!

    # Game variables
    colorIndex = (0, 0, 255)
    countRed = 0
    countCrack = 0
    countPath = 0
    pointsCut = []  # Stores all values of index finger

    # Load the path of the mission target image
    with open('path', 'rb') as f:
        pathMain = pickle.load(f)
    pointCrossList = [0] * len(pathMain)
    pathMainNP = np.array(pathMain, np.int32).reshape((-1, 1, 2))
    pathOuter = makeOffsetPoly(pathMain, difficulty)
    pathOuterNP = np.array(pathOuter, np.int32).reshape((-1, 1, 2))
    pathInner = makeOffsetPoly(pathMain, -difficulty)
    pathInnerNP = np.array(pathInner, np.int32).reshape((-1, 1, 2))

    # Sounds
    pygame.mixer.init()
    soundShot = pygame.mixer.Sound('./sounds/shot.mp3')
    soundCrack = pygame.mixer.Sound('./sounds/crack.wav')
    pygame.mixer.music.load("./sounds/timer.mp3")

    # Main loop
    start = True
    while start:
        # Get Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                start = False
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    exit()
                if event.key == pygame.K_s:
                    gameStart = True
                    soundShot.play()
                    pygame.mixer.music.play()

        # Apply Logic

        # OpenCV
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands = detector.findHands(img, draw=False)

        if gameOver is False and gameWon is False:

            img = cv2.addWeighted(img, 0.35, imgCookie, 0.9, 0)

            # Display the paths
            img = cv2.polylines(img, [pathMainNP], True, (225, 225, 225), 5)
            # img = cv2.polylines(img, [pathOuterNP], True, (0, 200, 0), 5)
            # img = cv2.polylines(img, [pathInnerNP], True, (0, 200, 0), 5)

            if hands:
                hand = hands[0]
                index = hand['lmList'][8][0:2]

                if gameStart:

                    # Check if index finger is inside the region
                    resultOutter = cv2.pointPolygonTest(pathOuterNP, index, False)
                    resultInner = cv2.pointPolygonTest(pathInnerNP, index, False)

                    # print(resultOutter,resultInner)

                    # When index in the correct region
                    if resultOutter == 1 and resultInner == -1:
                        # print("Inside")
                        colorIndex = (0, 255, 0)
                        countRed = 0
                    else:
                        colorIndex = (0, 0, 255)
                        countRed += 1
                        if countRed > 3:
                            gameStart = False
                            gameOver = True
                            print("Outside")
                            soundCrack.play()
                            pygame.mixer.music.stop()

                    # Check how many points/lines  have been passed
                    pointsCut.append(index)
                    if isPointInLine([pathOuter[countPath], pathInner[countPath]], pointsCut[-1]):
                        pointCrossList[countPath] = 1
                        countPath += 1

                    # cv2.line(img, pathOuter[countPath], pathInner[countPath], (0, 0, 255), 5)

                    if len(set(pointCrossList[:-2])) == 1 and pointCrossList[0] != 0:
                        print("gameWon")
                        gameWon = True
                        pygame.mixer.music.stop()

                    # Draw the path that has been covered
                    for x in range(1, len(pathMain)):
                        if pointCrossList[x] == 1:
                            cv2.line(img, pathMain[x - 2], pathMain[x - 1], (0, 200, 0), 10)
                    print(pointCrossList)
                cv2.circle(img, index, 10, colorIndex, cv2.FILLED)

        elif gameOver and gameWon is False:
            countCrack += 1

            # delay before playing shot sound
            if countCrack == 20:
                soundShot.play()
            # delay before changing the image
            if countCrack > 50:
                img = imgGameOver
            else:
                img = cv2.addWeighted(img, 0.35, imgCookieCrack, 0.9, 0)

        elif gameOver is False and gameWon:
            img = imgGameWon

        # Display image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB = np.rot90(imgRGB)
        frame = pygame.surfarray.make_surface(imgRGB).convert()
        frame = pygame.transform.flip(frame, True, False)
        window.blit(frame, (0, 0))

        # Update Display
        pygame.display.update()
        # Set FPS
        clock.tick(fps)


if __name__ == "__main__":
    Game()
