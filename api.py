from fastapi import FastAPI
from pydantic import BaseModel
import base64
from recognition import FaceRecognition
from utils import getBytes
face_recogniser = FaceRecognition()
app = FastAPI()

class Picture(BaseModel):
    Image: str

class UserInfo(BaseModel):
    Name: str
    Image: str


@app.post('/userinfo')
async def upload_info_endpoint(userinfo: UserInfo):
    img_data = getBytes(userinfo.Image)
    # with open("imageToSave.png", "wb") as fh:
    #     fh.write(base64.decodebytes(img_data))
    face_recogniser.add_face(img_data)

    return userinfo

@app.post('/picture')
async def get_faces(picture: Picture):
    img_data = getBytes(picture.Image)
    person = face_recogniser.run_recognition(img_data)
    return person
