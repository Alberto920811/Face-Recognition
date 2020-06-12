import os
import csv
import numpy as np 
import cv2.cv2 as cv2
import csv
import pandas as pd

#Utilizar este comando
#C:\Users\xcad\Desktop\FaceEmotionID\FaceEmotion_ID\dataimages>
# dir /s /b > print.txt
with open('dataimages/print.txt', 'r') as in_file:
    stripped = (line.split() for line in in_file)
    lines = (line for line in stripped if line)
    with open('dataset/paths.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        #writer.writerow('Path')
        writer.writerows(lines)

with open('dataset/paths.csv', 'r') as f:
  reader = csv.reader(f)
  first_list = list(reader)

#for item in your_list:

second_list = []
index = 0
for item in first_list:
    i = 2*index
    second_list.append(first_list[i][0])

third_list = []
index = 0
for item in second_list:
    third_list.append(item.replace("C:\\Users\\xcad\\Desktop\\FaceEmotionID\\FaceEmotion_ID\\",""))

pixeles = []
j = 0 

for item in third_list:
    w, h = 48, 48
    img = cv2.imread(third_list[j],0)
    res = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    res = res.flatten()
    pixeles.append(res)
    j += 1

my_df = pd.DataFrame(pixeles)
my_df.to_csv('dataset/pixeles.csv', index=False, header=False)


