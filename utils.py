import base64

def converter(img_b64):
    return base64.decodebytes(img_b64)

def saveImg(img, counter, path):
    with open(f"{path}/{counter}.jpg", "wb") as fh:
        fh.write(img)

def getBytes(img_str):
    return bytes(img_str, encoding="ascii")