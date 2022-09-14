import numpy as np
import pickle
import cv2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

testing_mode = True
random_state = 27

data =  pickle.load(open("data.pickle",'rb'))
X, Y = data
X = np.array(X)
Y = np.array(Y)
size = X.shape[1:]
print(size)

print(f"X shape {X.shape}")
print(f"Y shape {Y.shape}")

X = X/255.0
X_flat = X.reshape(X.shape[0], -1)
print(X_flat.shape)

if testing_mode:
    test_size = 0.4
    X_train, X_test, y_train, y_test = train_test_split(X_flat, Y, test_size=test_size, random_state=random_state, stratify=Y)
    print(f"n_classes {len(np.unique(Y))}")
    print(f"n_classes in y test {len(np.unique(y_test))}")
    print(f"n_classes in y train {len(np.unique(y_train))}")

else:
    X_train = X_flat
    y_train = Y

mean = np.mean(X_train, axis=0)
mean_face = np.reshape(mean, (100, 100))
pickle.dump(mean_face, open("mean_face.pkl","wb"))

pca = PCA(0.95).fit(X_train)
print(pca.n_components_)

pickle.dump(pca, open("pca_traning.pkl","wb"))

train_img_pca = pca.transform(X_train)
pca_dict = {}
for i in range(len(X_train)):
    label = y_train[i]
    pca_image = train_img_pca[i]

    if label in pca_dict:
        pca_dict[label].append(pca_image)
    else:
        pca_dict[label] = [pca_image]
pickle.dump(pca_dict, open("pca_dict.pkl","wb"))
