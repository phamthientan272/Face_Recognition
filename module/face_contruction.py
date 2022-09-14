import pickle
import numpy as np
import cv2
class MaskRemover:
    def __init__(self, mask_distance_from_eye=2, skin_threshold_weight=3, mask_threshold_weight=7, face_mask_size=(400, 400)):
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        self.mask_distance_from_eye = mask_distance_from_eye
        self.skin_threshold_weight = skin_threshold_weight
        self.mask_threshold_weight = mask_threshold_weight
        self.face_mask_size = face_mask_size

    def detect_eyes(self, img):
        eyes = self.eye_cascade.detectMultiScale(img)
        return eyes

    def get_skin_mask_val(self, img, eyes):
        if eyes[0][0] < eyes[1][0]:
            left_eye = eyes[0]
            right_eye = eyes[1]
        else:
            left_eye = eyes[1]
            right_eye = eyes[0]

        left_eye_top_right_corner = (left_eye[0] + left_eye[2], left_eye[1])
        right_eye_top_left_corner = (right_eye[0], right_eye[1])
        mid_eye = ((left_eye_top_right_corner[0] + right_eye_top_left_corner[0])//2 , right_eye_top_left_corner[1] )
        eye_distance = right_eye_top_left_corner[0] - left_eye_top_right_corner[0]

        skin_position = (mid_eye[0], mid_eye[1] - eye_distance)
        mask_position = (mid_eye[0], mid_eye[1] + self.mask_distance_from_eye*eye_distance)

        skin_val = img[skin_position]
        mask_val = img[mask_position]

        return skin_val, mask_val

    def bitwise_img(self, img):
        return cv2.bitwise_not(img)

    def focus_mask_frame(self, img, eyes, skin_val):
        bottom_eye = eyes[0][1] + eyes[0][3]
        mask_frame = img.copy()
        mask_frame[:bottom_eye, :] = 0
        return mask_frame

    def resize_image(self, img, size):
        return cv2.resize(img, size)

    def image_thresholding(self, img, skin_val, mask_val):
        thres_val = (mask_val*self.mask_threshold_weight + skin_val*self.skin_threshold_weight)//(self.mask_threshold_weight + self.skin_threshold_weight)
        _, mask_region = cv2.threshold(img, 90 , 1,cv2.THRESH_BINARY)
        return mask_region

    def remove_mask(self, img):
        original_size = img.shape
        img = self.resize_image(img, self.face_mask_size)
        eyes = self.detect_eyes(img)
        if len(eyes) == 2:
            skin_val, mask_val = self.get_skin_mask_val(img, eyes)
            mask_frame = self.focus_mask_frame(img, eyes, skin_val)
            if skin_val > mask_val:
                mask_frame = self.bitwise_img(mask_frame)
                skin_val, mask_val = self.get_skin_mask_val(mask_frame, eyes)
            mask_frame = self.resize_image(mask_frame, original_size)
            mask_region = self.image_thresholding(mask_frame, skin_val, mask_val)
            return mask_region
        else:
            return None

class FaceReconstruction():
    def __init__(self, pca_file: str, mean_face: str):
        self.mean_face = self.load_pickle(mean_face)
        self.faceshape = self.mean_face.shape
        self.pca = self.load_pickle(pca_file)
        self.eigenfaces = self.pca.components_[:]

    def load_pickle(self, pkl: str):
        return pickle.load(open(pkl,'rb'))

    def reconstruct(self, img):

        mask_region = self.extract_mask_region(img)  #mask pixel is 1
        if mask_region is None:
            return None

        mask_region_not = 1 - mask_region #mask pixel is 0

        img = cv2.resize(img, self.faceshape)
        img = img / 255.0

        mean_face_exclude_mask = self.exclude_mask(self.mean_face, mask_region_not)
        mean_face_include_mask = self.exclude_mask(self.mean_face, mask_region)

        eigenfaces_exclude_mask = [ self.exclude_mask(np.reshape(eigenface, self.faceshape), mask_region_not).flatten() for eigenface in self.eigenfaces ]

        eigenfaces_include_mask = [ self.exclude_mask(np.reshape(eigenface, self.faceshape), mask_region).flatten() for eigenface in self.eigenfaces ]


        test_face_exclude_mask = self.exclude_mask(img, mask_region_not)

        test_face_exclude_mask_substract_mean_face_exclude_mask = test_face_exclude_mask - mean_face_exclude_mask
        test_face_exclude_mask_substract_mean_face_exclude_mask = test_face_exclude_mask_substract_mean_face_exclude_mask.flatten()

        weights = self.get_weights(test_face_exclude_mask_substract_mean_face_exclude_mask, eigenfaces_exclude_mask)
        reconstructed_mask_region = self.reconstruct_mask_region(mean_face_include_mask, weights, eigenfaces_include_mask, mask_region)
        reconstructed_face = self.reconstruct_face(test_face_exclude_mask, reconstructed_mask_region)
        return reconstructed_face


    def extract_mask_region(self, img):
        mask_remover = MaskRemover()
        return mask_remover.remove_mask(img)

    def exclude_mask(self, img, mask):
        return np.multiply(img, mask)

    def get_weights(self, test_face_exclude_mask_substract_mean_face_exclude_mask, eigenfaces_exclude_mask):
        return [np.dot(eigenface, test_face_exclude_mask_substract_mean_face_exclude_mask) for eigenface in eigenfaces_exclude_mask]

    def reconstruct_mask_region(self, mean_face_include_mask, weights, eigenfaces_include_mask, mask_region):
        reconstructed_mask_region = mean_face_include_mask.flatten() + np.dot(weights, self.eigenfaces)
        reconstructed_mask_region = np.reshape(reconstructed_mask_region, self.faceshape)
        reconstructed_mask_region = self.exclude_mask(reconstructed_mask_region, mask_region)
        return reconstructed_mask_region

    def reconstruct_face(self, face_exclude_mask, reconstructed_mask_region):
        return (face_exclude_mask + reconstructed_mask_region)
