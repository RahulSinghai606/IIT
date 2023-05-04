# -*- coding: utf-8 -*-
"""q3_CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VdeODA0_zhJ86QrWU23p0kl9HG-CE7au

Question 3
"""

from keras.applications import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Set the input size of the images
img_width, img_height = 224, 224

# Set the directories of the training and validation data
train_data_dir = r"C:\Users\momo\Downloads\train_val-20230423T074600Z-001"
val_data_dir = r"C:\Users\momo\Downloads\train_val-20230423T074600Z-001"

# Create an instance of the VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add new layers to the pre-trained model
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Define the new model with the pre-trained model as its base and the new layers as its top
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with a binary crossentropy loss and an Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

# Set up data augmentation for the training data and validation data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

# Set the batch size
batch_size = 16

# Set the number of training and validation samples
nb_train_samples = 1600
nb_val_samples = 400

# Set the number of epochs
epochs = 10

# Train the model with the data generators
history = model.fit(
    train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary'),
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_datagen.flow_from_directory(val_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary'),
    validation_steps=nb_val_samples // batch_size)

# Evaluate the model on the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_data_dir =r"C:\Users\momo\Downloads\test-20230423T074543Z-001"

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)



