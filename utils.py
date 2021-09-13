import base64
import binascii


def decodeBase64(s):
    try:
        return base64.b64decode(s, validate=True)
    except binascii.Error:
        return False
