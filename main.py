import cv2
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, Activation
from tensorflow.python.client import device_lib
from keras.callbacks import EarlyStopping, ModelCheckpoint

print(device_lib.list_local_devices())

'''Block of code to extract every 15th frame from video'''
# count = 0
# y = 210
# x = 115
# h = 445
# w = 350
# dim = (116, 116)
# path = 'C:/Users/feuri/PycharmProjects/RevisorVision/Camera 3_20220526_003249.mp4'
# cap = cv2.VideoCapture()
# cap.open(path)
# frames = cap.get(cv2.CAP_PROP_FPS)
# print("frames=", int(frames))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = frame[y:y + h, x:x + w]
#         frame = cv2.resize(frame, dim)
#         cv2.imwrite(r"C:/Users/feuri/PycharmProjects/RevisorVision/Images/" + 'frame{:d}.jpg'.format(count), frame)
#         count += 15
#         cap.set(cv2.CAP_PROP_POS_FRAMES, count)
#     else:
#         cap.release()
#         break
#         print('Parsing has ended')

'''Because we have enough data to train and test our models,in order to not exceed the memory limit,
we'll be downloading our dataset by flows from directory by batches'''

train_gen = ImageDataGenerator(rescale=1. / 255)
valid_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.4)

train_data = train_gen.flow_from_directory(directory='C:/Users/feuri/PycharmProjects/RevisorVision/Images', seed=12345,
                                           target_size=(256, 256), shuffle=True, class_mode='binary')

valid_data = valid_gen.flow_from_directory(directory='C:/Users/feuri/PycharmProjects/RevisorVision/Images', seed=12345,
                                           target_size=(256, 256), shuffle=True,
                                           subset='training', class_mode='binary')

test_data = valid_gen.flow_from_directory(directory='C:/Users/feuri/PycharmProjects/RevisorVision/Images', seed=12345,
                                          target_size=(256, 256), shuffle=False,
                                          subset='validation', class_mode='binary')

CLASSES = train_data.class_indices
print(CLASSES)

# features, target = next(train_data)
#
# fig = plt.figure(figsize=(10, 10))
# for i in range(16):
#     fig.add_subplot(4, 4, i + 1)
#     plt.imshow(features[i])
#     plt.xticks([])
#     plt.yticks([])
#     plt.tight_layout()
# plt.show()
#
# '''Now let's add layers to our classification model'''
#
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# # here we use sigmoid function because we have only two choices and they have to be separate
# model.add(Activation('sigmoid'))
#
# '''Compiling the model with binary crossentrophy for the same reason'''
#
# cb = [EarlyStopping(patience=5, monitor='val_accuracy', mode='max', restore_best_weights=True),
#       ModelCheckpoint("RevisorVision.h5", save_best_only=True)]
#
# model.compile(loss='binary_crossentropy',
#               optimizer=keras.optimizers.Adam(1e-5),
#               metrics=['accuracy'])
#
# with tf.device('/gpu:0'):
#     model.fit(
#         train_data,
#         validation_data=val_data, callbacks=cb,
#         epochs=10, batch_size=32)

'''Now we are going to predict class of test data'''
model = keras.models.load_model('RevisorVision.h5')
y_prob = model.predict(test_data)
results = []
y_class =[]
for i in range(len(y_prob)):
    #y_class.append(y_prob[i].argmax(axis=-1))
    if y_prob[i] < .5:
        results.append('Absent') #100 - (100 * y_prob[i]))
    elif y_prob[i] > .5:
        results.append('At work') #y_prob[i] * 100)
    else:
        results = results.append('Not Sure')
print(results)
print(model.evaluate(test_data))
