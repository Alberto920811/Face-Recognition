from keras.preprocessing.image import img_to_array
import imutils
import cv2.cv2 
from keras.models import load_model
import numpy as np
import time
import csv
import datetime

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


# starting video streaming
#cv2.namedWindow('your_face')
num = 0
camera = cv2.VideoCapture(num)
n = []
tag = []
e_p = []
date = []
angry = []
disgust = [] 
scared = []
happy = []
sad = []
surprised = []
neutral = []
k = 0

while True:
    time.sleep(1)
    fecha = datetime.datetime.now()
    frame = camera.read()[1]

    #reading the frame
    frame = imutils.resize(frame,width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()

    if len(faces) > 0:

        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        #0 = enojado, 1 = asco, 2 = miedo, 3 = feliz,
        # 4 = triste, 5 = sorpresa, 6 = neutral 
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        n.append(num)
        tag.append(label)
        e_p.append(emotion_probability)
        date.append(str(fecha))
        angry.append(preds[0])
        disgust.append(preds[1])
        scared.append(preds[2])
        happy.append(preds[3])
        sad.append(preds[4])
        surprised.append(preds[5])
        neutral.append(preds[6])

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),(w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 255, 255), 2)
            cv2.putText(frameClone, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)
    
    elif len(faces) == 0:
        
        preds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        label = 'No detectado'
        emotion_probability = 0.0
        n.append(num)
        tag.append(label)
        e_p.append(emotion_probability)
        date.append(str(fecha))
        angry.append(preds[0])
        disgust.append(preds[1])
        scared.append(preds[2])
        happy.append(preds[3])
        sad.append(preds[4])
        surprised.append(preds[5])
        neutral.append(preds[6])
        

        cv2.imshow('your_face', frameClone)
    
    else: continue
    
    dict = {'Fecha': date[k], 'Camara': n[k], 'Etiqueta': tag[k], 'Confianza': e_p[k],
        "angry": angry[k] ,"disgust": disgust[k],"scared": scared[k], "happy": happy[k], 
        "sad": sad[k], "surprised": surprised[k],"neutral": neutral[k]}
    with open('dataset/output_data.csv', 'a') as csvfile:
        fieldnames = ['Fecha', 'Camara', 'Etiqueta', 'Confianza',"angry" ,"disgust",
        "scared", "happy", "sad", "surprised","neutral"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict)
    
    k += 1
    print(k)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()