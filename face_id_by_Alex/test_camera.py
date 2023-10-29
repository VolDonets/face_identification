import numpy as np
import cv2
from deepface import DeepFace
from deepface.detectors import FaceDetector


def face_verify(img_1, img_2):
    try:
        result1 = DeepFace.verify(img1_path=img_1, img2_path=img_2, model_name="Facenet")
        return result1['verified']
    except Exception as _ex:
        return False


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

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

        db = ("anton.jpg", "goat.jpg", "obama.jpg", "patron.jpg", "roshen.jpg", "the_kingo.jpg")
        db_res = [face_verify(img_1=face_[0], img_2=f'../face_id_by_Anton/data/{known_face}') for known_face in db]

        text = db[db_res.index(True)] if True in db_res else f"UNDEFINED_{inx_}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y)
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 2
        img = cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
