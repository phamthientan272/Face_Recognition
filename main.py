import numpy as np
import cv2
import pickle
from facenet_pytorch import MTCNN
from module.make_detection import MaskDetection
from module.recognizer import Recognizer_NN,  Recognizer_Euclidean
from module.face_contruction import FaceReconstruction
from module.face_alignment import FaceAlignment

import torch

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


recognizer_type = "ED"
if recognizer_type == "NN":
    model_path = 'saved_model/dense_network'
    recognizer = Recognizer_NN(model_path)
    pca_file = "saved_model/pca_traning.pkl"
    pca = pickle.load(open(pca_file,'rb'))
elif recognizer_type == "ED":
    model_path = 'saved_model/pca_dict.pkl'
    recognizer = Recognizer_Euclidean(model_path)
    pca_file = "saved_model/pca_traning.pkl"
    pca = pickle.load(open(pca_file,'rb'))

labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


mean_face_file = "mean_face.pkl"
face_reconstruction = FaceReconstruction(pca_file=pca_file, mean_face=mean_face_file)

font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)
stroke = 2
while True:
    try:
        ret, frame = cap.read()
        try:

            boxes, _, points = mtcnn.detect(frame, landmarks=True)
            eyes = points[0][:2]
        except:
            pass

        if boxes is not None:
            for (box, point) in zip(boxes, points):
                bbox = list(map(int,box.tolist()))
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                roi_color = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                roi_color = FaceAlignment().align_img(roi_color, eyes)
                roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)


                is_mask = MaskDetection().is_mask(roi_gray)
                if is_mask is False:
                    image = cv2.resize(roi_gray, (100, 100))
                    image = cv2.equalizeHist(image)
                    image = image / 255.0
                    image = np.reshape(image, (1, -1))
                    pca_image = pca.transform(image)

                    id, confidence = recognizer.predict(pca_image)
                    print("confidence ", confidence)

                    if recognizer_type == "NN":
                        THRESHOLD = 0.6
                        if confidence >= THRESHOLD:
                            name = labels[id]
                            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                        else:
                            name = "Unknown"
                            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

                    if recognizer_type == "ED":
                        THRESHOLD = 20
                        if confidence <= THRESHOLD:
                            name = labels[id]
                            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                        else:
                            name = "Unknown"
                            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                else:

                    image = cv2.resize(roi_gray, (100, 100))
                    image = cv2.equalizeHist(image)
                    reconstructed_face = face_reconstruction.reconstruct(image)
                    if reconstructed_face is None:
                        name = "Unknown"
                        cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                    else:
                        print("reconstructed")
                        image = np.reshape(reconstructed_face, (1, -1))
                        pca_image = pca.transform(image)
                        id, confidence = recognizer.predict(pca_image)
                        print("confidence ", confidence)
                        if recognizer_type == "NN":
                            THRESHOLD = 0.8
                            if confidence >= THRESHOLD:
                                name = labels[id]
                                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                            else:
                                name = "Unknown"
                                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

                        if recognizer_type == "ED":
                            THRESHOLD = 25
                            if confidence <= THRESHOLD:
                                name = labels[id]
                                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                            else:
                                name = "Unknown"
                                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

                color = (255, 0, 0)
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    except:
        print("error")

cap.release()
cv2.destroyAllWindows()

