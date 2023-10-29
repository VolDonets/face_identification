import numpy as np
import pandas as pd

import cv2

from deepface import DeepFace
from deepface.detectors import FaceDetector

import DeepFace_custom


faces_df = pd.read_csv("./data_embeddings/data_embeddings_by_Facenet.csv")


def verify_face_name(img_1):
    try:
        img_embed = DeepFace_custom.to_embedding(img_1, model_name="Facenet")[0]["embedding"]
    except:
        return "UNDEFINED"
    for face_name in faces_df.columns:
        if DeepFace_custom.veriby_by_embeddins(img_embed, faces_df[face_name].to_numpy(), model_name="Facenet"):
            return face_name

    return "UNDEFINED"


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = FaceDetector.build_model('opencv')

while True:
    ret, img = cap.read()
    if not ret:
        print("Unable to read image")

    faces = FaceDetector.detect_faces(detector, "opencv", img)

    for inx_, face_ in enumerate(faces):
        x, y, w, h = face_[1]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_color = img[y:y + h, x:x + w]

        text = verify_face_name(roi_color)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y)
        font_scale = 0.5
        color = (165, 255, 0)
        thickness = 2
        img = cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
