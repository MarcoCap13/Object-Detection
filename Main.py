import easygui
from easygui import *
from ObjectDetection import objectDetection, photoDetection
import cv2


'''
Progetto Machine Learning 2022
studenti: Caponi Marco & Ceneda Gianluca

Programma che permette di riconoscere oggetti attraverso l'utilizzo della videocamera o di un file jpeg/jpg.
Abbiamo utilizzato una libreria (ristretta) chiamata "Tiny YOLOv3" per poter lavorare con un numero ristretto di elementi
Per finire, per ogni oggetto riconosciuto, il programma ci stampa le info. principali attraverso l'utilizzo della liberia
'wikipediaapi'
'''
# Ulteriori sviluppi:
'''
Durante il corso di deep learning pensiamo di aggiornare il progetto lavorando sull'acquisizione dei dati presi direttamente dal 
programma. 

'''

#parte grafica per la scelta dell'opzione
msg = "Vuoi eseguire l'image detection dalla videocamera o attraverso una foto?"
title = "Image Detection"
button_list = []
# first button
button1 = "Video detection"
# second button
button2 = "Photo detection"

button_list.append(button1)
button_list.append(button2)
output = buttonbox(msg, title, button_list)
#print(output)

if output == "Video detection":  # mostra un dialogo che consente all'utente di scegliere l'opzione desiderata tra Video detection e Photo detection
    objectDetection()
elif output == "Photo detection":
    photoDetection()
#else:  # user chose Cancel <-- implementazione futura per la chiusura della gui se l'utente non sceglie nessuna opzione
    #easygui.sys.exit(0)

