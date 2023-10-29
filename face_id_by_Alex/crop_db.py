import cv2
from deepface.detectors import FaceDetector


detector = FaceDetector.build_model('opencv')

db = ("anton.jpg", "goat.jpg", "obama.jpg", "patron.jpg", "roshen.jpg", "the_kingo.jpg")
images = [cv2.imread(f'../face_id_by_Anton/data/{known_face}') for known_face in db]

for jnx_, img in enumerate(images):
    faces = FaceDetector.detect_faces(detector, "opencv", img)

    for inx_, face_ in enumerate(faces):
        x, y, w, h = face_[1]
        roi_color = img[y:y + h, x:x + w]
        cv2.imwrite(f'../face_id_by_Alex/data_cropped/{db[jnx_]}', roi_color)
