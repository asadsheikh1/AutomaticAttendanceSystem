import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'training'
images = list()
person_names = list()
person_list = os.listdir(path)

for cu_image in person_list:
    current_image = cv2.imread(f'{path}/{cu_image}')
    images.append(current_image)
    person_names.append(os.path.splitext(cu_image)[0])

print(person_list)
print(person_names)


def face_encodings(_images):
    encode_list = []
    for image in _images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encode_list.append(encode)
    return encode_list


def attendance(_name):
    with open(f'Attendance.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []

        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])

        if _name not in name_list:
            time_now = datetime.now()
            time_string = time_now.strftime('%H:%M:%S')
            date_string = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{time_string},{date_string}')


encode_list_known = face_encodings(images)
print('Encodings completed.')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    faces_current_frame = face_recognition.face_locations(faces)
    encodes_current_frame = face_recognition.face_encodings(faces, faces_current_frame)

    for encode_face, faceLoc in zip(encodes_current_frame, faces_current_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)
        matchIndex = np.argmin(face_dis)

        if matches[matchIndex]:
            name = person_names[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 215), 3)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 215), cv2.RETR_FLOODFILL)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Automated Attendance | Camera', frame)

    keys = cv2.waitKey(1) & 0xFF
    if keys == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
