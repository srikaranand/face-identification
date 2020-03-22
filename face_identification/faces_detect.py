import numpy as np
import face_recognition as fr
import os
import cv2
from random import randint as rand

def initiate_data():
    data=dict()
    fnames = [f for f in os.listdir("data//") if os.path.isfile(os.path.join("data//", f))]
    for fname in fnames:
        name = fname.split('.')
        name=name[0]
        image = fr.load_image_file(os.path.join("data//",fname))
        face = fr.face_encodings(image)[0]
        data[name]=face
    return data

def find_faces_present(image):
    faces_present = fr.face_encodings(image)
    faces_locations = fr.face_locations(image)
    return faces_locations

def sort_faces_present(face_locations):
    face_locations_left = [tup[3] for tup in face_locations]
    zipped_lists = zip(face_locations_left,face_locations)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    face_locations_left,face_locations = [ list(tuple) for tuple in  tuples]
    return face_locations_left,face_locations

def match_faces(data,face_locations,image):
    names_present = []
    faces_present = fr.face_encodings(image, face_locations)
    names=list(data.keys())
    known_faces=list(data.values())
    for unknown_face in faces_present:
        result = fr.compare_faces(known_faces,unknown_face)
        if True in result:
            names_present.append(names[result.index(True)])
        else:
            names_present.append('unknown')
    return names_present

def show_faces(image,face_locations,names):
    for (top, right, bottom, left), name in zip(face_locations, names):
        cv2.rectangle(image, (left-10, top-10), (right+10, bottom+10), (rand(0,256), rand(0,256), rand(0,256)), 2)
        cv2.rectangle(image, (left-10, bottom -10), (right+10, bottom+10), (rand(0,256), rand(0,256), rand(0,256)), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, name, (left -10, bottom + 10), font, 0.5, (255, 255, 255), 2)
    while True:
        cv2.imshow('Detected Faces', image)
        if cv2.waitKey(0):
            return

def printlist(names_present):
    print('Faces Present : ')
    for i in range(len(names_present)):
        print(names_present[i],end='')
        if i<len(names_present)-1:
            print(', ',end='')

data = initiate_data()
fn = 'test.jpg'
image = fr.load_image_file(fn)
faces_locations = find_faces_present(image)
face_locations_left,face_locations = sort_faces_present(faces_locations)
names_present = match_faces(data,face_locations,image)
printlist(names_present)
images = cv2.imread(fn, 1)
show_faces(image,face_locations,names_present)