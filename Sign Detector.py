# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 23:32:10 2022

@author: tooba_29
"""
from PIL import Image
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2



# im = Image.open(r"C:\Users\square\Desktop\TOOBA\UNI\SEM6\ML\Final Project\Gesture Image Data/0/1.jpg")
# im.show()



# original_folder=r"C:\Users\square\Desktop\TOOBA\UNI\SEM6\ML\Final Project\Gesture Image Data/9"
# train_folder= r"C:\Users\square\Desktop\TOOBA\UNI\SEM6\ML\Final Project\train\Z"
# validate_folder=r"C:\Users\square\Desktop\TOOBA\UNI\SEM6\ML\Final Project\validate\9"
# test_folder= r"C:\Users\square\Desktop\TOOBA\UNI\SEM6\ML\Final Project\test\Z"

# # os.mkdir(train_folder)
# # os.mkdir(validate_folder)
# # os.mkdir(test_folder)

# filenames= [f"{i}.jpg" for i in range(1, 1201)]
# for filename in filenames:
#     src_path = os.path.join(original_folder, filename)
#     dst_path = os.path.join(train_folder, filename)
#     shutil.copy(src_path, dst_path)

# filenames= [f"{i}.jpg" for i in range(1051, 1201)]
# for filename in filenames:
#     src_path = os.path.join(original_folder, filename)
#     dst_path = os.path.join(validate_folder, filename)
#     shutil.copy(src_path, dst_path)

# filenames= [f"{i}.jpg" for i in range(1201, 1501)]
# for filename in filenames:
#     src_path = os.path.join(original_folder, filename)
#     dst_path = os.path.join(test_folder, filename)
#     shutil.copy(src_path, dst_path)


train_path= r"C:\Users\square\Desktop\TOOBA\UNI\SEM6\ML\Final Project\train"
validate_path= r"C:\Users\square\Desktop\TOOBA\UNI\SEM6\ML\Final Project\validate"
test_path= r"C:\Users\square\Desktop\TOOBA\UNI\SEM6\ML\Final Project\test"

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(64,64), class_mode='categorical', batch_size=10,shuffle=True)
validate_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=validate_path, target_size=(64,64), class_mode='categorical', batch_size=10,shuffle=True)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(64,64), class_mode='categorical', batch_size=10, shuffle=True)

imgs, labels = next(train_batches)
#Plotting the images...
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(30,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plotImages(imgs)
print(imgs.shape)
print(labels)


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
#model.add(Dropout(0.2))
#model.add(Dropout(0.3))
model.add(Dense(36,activation ="softmax"))
model.summary()



model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


history2 = model.fit(x=train_batches, epochs=10,  validation_data = validate_batches, verbose=1)
history2.history

# For getting next batch of validating imgs...
imgs_v, labels_v = next(validate_batches)  
scores_v = model.evaluate(imgs_v , labels_v , verbose=0) 
print(f'{model.metrics_names[0]} of {scores_v[0]}; {model.metrics_names[1]} of {scores_v[1]*100}%') 
# Once the model is fitted we save the model using model.save()  function.
model.save('best_model_dataflair2.h5')
word_dict_v = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z'}           

predictions_v = model.predict(imgs_v, verbose=0) 
print("predictions on a small set of validation data--") 
print("")
for ind, i in enumerate(predictions_v): 
    print(word_dict_v[np.argmax(i)], end='   ') 
plotImages(imgs)
print('Actual labels')
for i in labels:
    print(word_dict_v[np.argmax(i)], end='   ')
    
    
# For getting next batch of testing imgs...
imgs, labels = next(test_batches) 
scores = model.evaluate(imgs, labels, verbose=0)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
# Once the model is fitted we save the model using model.save()  function.
model.save('best_model_dataflair3.h5')

word_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z'}         

predictions = model.predict(imgs, verbose=0)
print("predictions on a small set of valid data--")
print("")
for ind, i in enumerate(predictions):
    print(word_dict[np.argmax(i)], end='   ')
plotImages(imgs)
print('Actual labels')
for i in labels:
    print(word_dict[np.argmax(i)], end='   ')    
    
    
    

   
    