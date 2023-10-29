import numpy as np
import pandas as pd
import imutils
import cv2
from deepface.detectors import FaceDetector

# 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'
tracker_type_ = 'KCF'


# def calculate_overlap_percentage(roi1, roi2):
#     # roi == np.array([[x0, y0], [x1, y1]])
#
#     # Calculate intersection coordinates
#     x_min = max(roi1[0, 0], roi2[0, 0])
#     y_min = max(roi1[0, 1], roi2[0, 1])
#     x_max = min(roi1[1, 0], roi2[1, 0])
#     y_max = min(roi1[1, 1], roi2[1, 1])
#     print(x_min, x_max, "-", y_min, y_max)
#
#     # Calculate width and height of overlap
#     width = max(0, x_max - x_min)
#     height = max(0, y_max - y_min)
#
#     # Calculate area of overlap
#     area_overlap = width * height
#
#     # Calculate total areas
#     area_roi1 = (roi1[1, 0] - roi1[0, 0]) * (roi1[1, 1] - roi1[0, 1])
#     area_roi2 = (roi2[1, 0] - roi2[0, 0]) * (roi2[1, 1] - roi2[0, 1])
#
#     return ((area_overlap / min(area_roi1, area_roi2)) + (area_overlap / max(area_roi1, area_roi2))) / 2

def calculate_overlap_percentage(roi1, roi2):
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2

    # Calculate intersection coordinates
    x_i = max(x1, x2)
    y_i = max(y1, y2)
    w_i = min(x1 + w1, x2 + w2) - x_i
    h_i = min(y1 + h1, y2 + h2) - y_i

    # Calculate area of overlap
    area_overlap = w_i * h_i

    # Calculate total areas
    area_roi1 = w1 * h1
    area_roi2 = w2 * h2

    return ((area_overlap / min(area_roi1, area_roi2)) + (area_overlap / max(area_roi1, area_roi2))) / 2


def get_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create

    return tracker


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = FaceDetector.build_model('opencv')

tracker_obj = None

frames_count = 0

while True:
    ret, img = cap.read()
    if not ret:
        print("Unable to read image")

    if not tracker_obj:
        faces = FaceDetector.detect_faces(detector, "opencv", img)

        for inx_, face_ in enumerate(faces):
            x, y, w, h = face_[1]

            tracker_obj = get_tracker(tracker_type_)()
            tracker_obj.init(img, face_[1])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            break
    else:
        tracked_res = tracker_obj.update(img)

        if tracked_res[0]:
            x, y, w, h = tracked_res[1]

            is_tracker_correct = True
            if frames_count % 25 == 0:
                frames_count = 0
                faces = FaceDetector.detect_faces(detector, "opencv", img)

                is_tracker_correct = False
                for face_ in faces:
                    overlap = calculate_overlap_percentage(face_[1], tracked_res[1])
                    print(overlap)
                    if overlap > 0.7:
                        is_tracker_correct = True
                        break

            if is_tracker_correct:
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
            else:
                tracker_obj = None
                print("Loose tracker by FaceDetector")
        else:
            tracker_obj = None
            print("Loose tracker")

    frames_count += 1
    cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
