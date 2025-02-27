import cv2 
import numpy as np
import face_recognition
import os

path = "Persons"
images = []
classNames = []
# for take all images from the path
personList = os.listdir(path) 
# print(personList)

for person in personList:
    curPerson = cv2.imread(f'{path}/{person}')
    images.append(curPerson)
    # take ony name without extention .jpg .png ...etc
    classNames.append(os.path.splitext(person)[0])
print(classNames)

def findEncodings(image):
    encodeList = []
    for img in images:
        # change colors to rgb cz face_reco needs rgb images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # make some points on the face on the image to detect it
        encode = face_recognition.face_encodings(img)

        if encode :
            encodeList.append(encode[0])
    return encodeList

encodeListKnown = findEncodings(images)
print(encodeListKnown)
print("encoding complete.")

# use laptop camera ? 0 : 1
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    # make the size of image small for fast running
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # draw a sqyare on the face and write the name
    faceCurentFrame = face_recognition.face_locations(imgS)
    encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)

    for encodeface, faceloc in zip(encodeCurentFrame, faceCurentFrame):
        # compare the known faces with faces on video
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        # numbers less number for the person on video
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        # take the less number on faceDis
        matchIndex = np.argmin(faceDis)

        # if the person found on the list
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            # color an type of font
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        else:
            name2 = "Not Known"
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)
            # color an type of font
            cv2.putText(img, name2, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    # use video ? 1 || camera ? 0
    cv2.imshow('Face Recognition', img)
    cv2.waitKey(1)