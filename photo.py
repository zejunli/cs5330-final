import cv2
from cv2 import COLOR_BGR2GRAY
from preprocessor import Network, HandGestureDataset
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from time import time


label_mapping = {
    0: 'down', 1: 'palm', 2: 'I', 3: 'fist',
    4: 'fist_moved', 5: 'thumb', 6: 'index', 7: 'ok',
    8: 'palm_moved', 9: 'c'
}


def video():
    cap = cv2.VideoCapture(0)
    print(cap.isOpened())
    
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            # frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, dsize=(640, 240))
            
            res, box = test_model(img=frame)
            cv2.putText(frame, res, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, 2)
            if box is not None:
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            cv2.imshow("Video", frame)

            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elif cv2.waitKey(1) & 0xFF == ord('s'):
                print('Photo name: ', end='')
                filename = input()
                cv2.imwrite(filename, frame)
                
        else:
            break
    
    cv2.destroyWindow('Video')
    return 


def dark_background():
    img = cv2.imread('./manual/I.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    key = cv2.waitKey(1)
    while key & 0xFF != ord('q'):
        cv2.imshow('hello1', hsv)
        mask = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))
        mask = cv2.erode(cv2.dilate(mask, None), None)
        cv2.imshow('hello', mask)
        output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

        max_area = 0
        max_label = 0
        stats = output[2]
        for i in range(1, output[0]):
            line = stats[i]
            if line[-1] > max_area:
                max_area = line[-1]
                max_label = i

        
        mask[output[1] != max_label] = 0
        
        # draw bbox
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        rect = cv2.minAreaRect(cnts[0])
        box = np.int0(cv2.boxPoints(rect))
        cv2.imshow("Cleaned", mask)

        combined = cv2.bitwise_and(img, img, mask=mask)
        cv2.drawContours(combined, [box], 0, (0, 0, 255), 2)
        
        cv2.imshow("hello2", combined)
        key = cv2.waitKey(1)

    for c in output[2]:
        print(c)

    return 


def test_model(img=None):
    # load data
    if img is None:
        img = cv2.imread('./manual/palm.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))
    mask = cv2.erode(cv2.dilate(mask, None), None)
    output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    max_area = max_label = -1
    stats = output[2]
    for i in range(1, output[0]):
        line = stats[i]
        if line[-1] > max_area:
            max_area = line[-1]
            max_label = i

    
    mask[output[1] != max_label] = 0

    # draw bbox
    box = None
    
    if max_area != -1:
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        rect = cv2.minAreaRect(cnts[0])
        box = np.int0(cv2.boxPoints(rect))
    
    combined = cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask), cv2.COLOR_BGR2GRAY)
    combined = cv2.resize(combined, (128, 48))



    test_ds = HandGestureDataset(np.array([combined]), np.array([2]), training=False, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]))

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True)

    network = Network()
    # load train network
    network.load_state_dict(torch.load('./model.pt'))

    
    
    network.eval()
    t_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            prediction = output.data.max(1, keepdim=True)[1]
            return label_mapping[prediction.numpy()[0][0]], box


if __name__ == '__main__':
    video()
    # dark_background()
    # test_model()


