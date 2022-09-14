import cv2
import numpy as np

class FaceAlignment:
    def __init__(self):
        pass

    def euclidean_distance(self, a, b):
        x1 = a[0]; y1 = a[1]
        x2 = b[0]; y2 = b[1]
        return np.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

    def align_img(self, img, eyes):
        eye_1 = eyes[0]
        eye_2 = eyes[1]
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        left_eye_x = left_eye[0]
        left_eye_y = left_eye[1]
        right_eye_x = right_eye[0]
        right_eye_y = right_eye[1]

        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 #rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 #rotate inverse direction of clock

        a = self.euclidean_distance(left_eye, point_3rd)
        b = self.euclidean_distance(right_eye, point_3rd)
        c = self.euclidean_distance(right_eye, left_eye)

        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a)
        angle = (angle * 180) / np.pi

        if direction == -1:
            angle = 90 - angle

        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1

        M = cv2.getRotationMatrix2D(center, direction * angle, scale)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated
