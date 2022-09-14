import cv2

class MaskDetection:
    def __init__(self, scaleFactor=1.5, minNeighbors=20, size=(400, 400), mouth_haar_path='haarcascade_mcs_mouth.xml'):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.size = size
        self.mouth_cascade = cv2.CascadeClassifier(mouth_haar_path)

    def resize_img(self, img):
        return cv2.resize(img, self.size)

    def is_mask(self, img):
        img = self.resize_img(img)
        mouths = self.mouth_cascade.detectMultiScale(img, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors)
        if len(mouths) > 0:
            return False
        else:
            return True
