import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image


feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3),pooling='avg')  # 'avg' pooling for feature extraction

detector = MTCNN()
# load img -> face detection
sample_img = cv2.imread('sample_photos/sallu.jpg')
results = detector.detect_faces(sample_img)

x,y,width,height = results[0]['box']

face = sample_img[y:y+height,x:x+width]


# extract its features
image = Image.fromarray(face)
image = image.resize((224,224))

face_array = np.asarray(image)

face_array = face_array.astype('float32')

expanded_img = np.expand_dims(face_array,axis=0)
preprocessed_img = preprocess_input(expanded_img)
result = model.predict(preprocessed_img).flatten()
print(result)
print(result.shape)
# find the cosine distance of current image with all the 8664 features
similarity = []
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])

index_pos = sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]#similarity with 8664th image will come first

temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output',temp_img)
cv2.waitKey(0)
# recommend that image





# load image ->face detectio and extract its features
# find the cosine distance of current image with all the 8664 features
# recommend that image