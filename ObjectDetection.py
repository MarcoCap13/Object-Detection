import cv2
import time, os, tensorflow as tf
from matplotlib import image
import numpy as np
import tensorflow
import wikipediaapi
import sys
from contextlib import redirect_stdout



def objectDetection():
        
    thres = 0.5 #  soglia per il riconoscimento

    #SETTIGS per la videocamera   
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    cap.set(10,70)

    #dataset
    classNames= []
    classFile = 'Labels.txt'

    with open(classFile,'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')


    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
        
    while True: 
        success,img = cap.read()
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        print(classIds,bbox)

        # se rileviamo qualcosa..
        if len(classIds) != 0:
            # per ogni info. rilevata..
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                if (confidence*100 < 56):
                    cv2.rectangle(img,box,color=(0,0,255),thickness=2)                          # creiamo un rettangolo
                    #nome dell'oggetto in camera
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    #Affidabilita
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

                else:
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)                          # creiamo un rettangolo
                    #nome dell'oggetto in camera
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    #Affidabilita
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

                '''
                    - implementare che se la confidenze < n --> bbox colore rosso, sennò bbox colore verde
                    - 
                '''
                #Per wikipedia
                #wiki_wiki = wikipediaapi.Wikipedia('en')
                #page_py = wiki_wiki.page(classNames[classId-1].lower())
                #cv2.putText(img, page_py.summary[0:70],(box[0]+50, box[1]+70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0),2)
                
                #facciamo la parte grafica con easygui. "pip install easygui". Cosi possiamo fare anche la parte grafica ed
                #e abbastanza facile. Cosi leviamo la console e l'utente puo decidere cosa selezioanre direttamente con un bottone

                #stampa su prompt
                #print (page_py.summary[0:60])         
        
            cv2.imshow("Output",img)
            cv2.waitKey(1)

        # quando premeremo 0 il video si stopperà..
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

def photoDetection():
# NO: macchina1, persone1,
    img = cv2.imread('test/test1.jpeg')
    classNames = []
    classFile = 'Labels.txt'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
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
        page = classNames[classId-1]
        print(page_py.summary + "\n")
        descrizione = page_py.summary

        

        # Scrittura su file markdown
        with open("test.md", "a+") as f:        # a+ crea un file di scrittura e lettura
            data = f.read(100)                  # se il dato è esistente..
            if len(data) >0:                    # e la lunghezza è diversa da 0
                f.write("\n")                   
            f.write(descrizione + "\n")         # scrivici sopra
            f.write("=========================================================================\n\n")

    cv2.imshow("Output", img)
    cv2.waitKey(0)
    open('test.md', 'a+').truncate()
