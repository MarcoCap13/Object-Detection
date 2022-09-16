import cv2
import wikipediaapi 
import easygui
import subprocess
import test as test
import tensorflow


def photoDetection():
    #Carico l'immagine desiderata dalla cartella "test"
    img = cv2.imread('test/test2.jpeg')

    classNames = []
    classFile = 'Labels.txt'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'         #Dataset ristretto YOLO
    weightsPath = 'frozen_inference_graph.pb'

    
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)

    for classId, confidence , box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        # Affidabilita
        cv2.putText(img, classNames[classId-1].upper(),(box[0] + 10, box[1]+30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0),1)
        # per wikipedia
        wiki_wiki = wikipediaapi.Wikipedia('en') #uso la libreria in inglese

        page_py = wiki_wiki.page(classNames[classId-1].lower())
        #Stampalo nel bbox
        cv2.putText(img, page_py.summary[0:60],(box[0]+50, box[1]+70), cv2.FONT_HERSHEY_PLAIN, 1, (22, 44, 100),1)
        #stampalo su linea di comando        
    cv2.imshow("Output", img)
    cv2.waitKey(0)

