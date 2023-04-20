import base64

def converter(img_b64):
    return base64.decodebytes(img_b64)


def getBytes(img_str):
    return bytes(img_str, encoding="ascii")