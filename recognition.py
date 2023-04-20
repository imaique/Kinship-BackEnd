import face_recognition
import os
import boto3
import cv2
import numpy as np
import math
from PIL import Image
from utils import converter
from db_compare import db_compare
from pathlib import Path

AWS_BUCKET = 'kinship-bucket'


session = boto3.Session(
    aws_access_key_id='AKIASMMC5NREJZCNV3NL',
    aws_secret_access_key='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'
)

s3 = session.resource('s3')
bucket = s3.Bucket(AWS_BUCKET)

some_binary_data = b'Here we have some data'
object = s3.Object(AWS_BUCKET, 'filename.jpg')
object.put(Body=some_binary_data)




bucket.put_object(Key='test.txt')

def uploadImg(img: bytes, counter, event):
    object = s3.Object(AWS_BUCKET, f'{counter}.jpg')
    object.put(Body=img)

def saveImg(img: bytes, counter, event):
    with open(f"{event}/{counter}.jpg", "wb") as binary_file:
        binary_file.write(img)
    
    

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + ' %'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + ' %'


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_ids = []
    counter = 0

    def __init__(self):
        self.encode_faces()

    def add_face(self, image_bytes):
        image_id = self.counter
        self.uploadImg(image_bytes, image_id, "polyhacks")
        self.saveImg(image_bytes, image_id, "polyhacks")
        self.counter += 1
        self.append_image(f'{image_id}.jpg')

    def append_image(self, image):
        face_image = face_recognition.load_image_file(f'faces/{image}')
        face_encoding = face_recognition.face_encodings(face_image)[0]

        self.known_face_encodings.append(face_encoding)
        self.known_face_ids.append(image)

    def encode_faces(self):

        for file_name in bucket.objects.filter(Prefix="polyhacks", Delimiter="/"):
            file_path = Path('polyhacks/' + file_name.key)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            print(file_path)
            #bucket.download_file(
            #    file_path,
            #    file_path
            #)
            self.counter += 1

        for image in os.listdir('faces'):
            self.append_image(image)
        

    def run_recognition(self, img_b64):
        img_bytes = converter(img_b64)
        self.saveImg(img_bytes,0,"temp")
        
        # numpydata = np.asarray()

        # small_frame = cv2.resize(numpydata, (0, 0), fx=0.25, fy=0.25)
        face_image = face_recognition.load_image_file(f'temp/0.jpg')

        small_frame = cv2.resize(face_image, (0, 0), fx=0.25, fy=0.25)

        # rgb_small_frame = small_frame[:, :, ::-1]

        # Find all faces in the current frame
        self.face_locations = face_recognition.face_locations(small_frame)  # detects faces
        print("self.face_locations = ",self.face_locations)
        self.face_encodings = face_recognition.face_encodings(small_frame, self.face_locations)
        print("self.face_encodings = ",self.face_encodings)

        self.face_names = []

        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = 'Unknown'
            confidence = 'Unknown'

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_ids[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])
            self.face_names.append(f'{name} ({confidence})')
        print("self.face_names == ",self.face_names)
        informations = db_compare(self.face_names,self.face_locations)
        return informations


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
