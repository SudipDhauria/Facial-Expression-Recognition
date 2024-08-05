import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = 'data/'  # Replace with your data directory
categories = ['happy', 'sad', 'angry', 'neutral']  # Example categories

IMG_SIZE = 48

def create_training_data():
    training_data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
    return training_data

training_data = create_training_data()
