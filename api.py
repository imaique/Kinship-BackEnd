from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from recognition import FaceRecognition
from utils import getBytes
face_recogniser = FaceRecognition()


origins = ["*"]

middleware = [
    Middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)]

app = FastAPI(middleware = middleware)


class Picture(BaseModel):
    Image: str

class UserInfo(BaseModel):
    Name: str
    Image: str

def trim(str):
    index = str.find(',')
    print(index)
    return str[index+1:]

@app.post('/userinfo')
async def upload_info_endpoint(userinfo: UserInfo):
    trimmed_string = trim(userinfo.Image)
    img_data = getBytes(trimmed_string)
    face_recogniser.add_face(img_data)

    return userinfo

@app.post('/picture')
async def get_faces(picture: Picture):
    trimmed_string = trim(picture.Image)
    img_data = getBytes(trimmed_string)
    person = face_recogniser.run_recognition(img_data)
    return person
