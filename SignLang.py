import os
from IPython.display import Image     
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data(filename):
    with open(filename) as training_file:
        training_reader = csv.reader(training_file, delimiter=',')
        image = []
        labels = []
        line_count = 0
        for row in training_reader:
            if line_count == 0:
                line_count +=1
            else:
                labels.append(row[0])
                temp_image = row[1:785]
                image_data_as_array = np.array_split(temp_image, 28)
                image.append(image_data_as_array)
                line_count += 1
        images = np.array(image).astype('float')
        labels = np.array(labels).astype('float')
        print(f'Processed {line_count} lines.')

    return images, labels


training_images, training_labels = get_data("dataset/sign_mnist_train/sign_mnist_train.csv")
testing_images, testing_labels = get_data("dataset/sign_mnist_test/sign_mnist_test.csv")

print("Total Training images", training_images.shape)
print("Total Training labels",training_labels.shape)
print("Total Testing images",testing_images.shape)
print("Total Testing labels",testing_labels.shape)


alphabets = 'abcdefghijklmnopqrstuvwxyz'
mapping_letter = {}

for i,l in enumerate(alphabets):
    mapping_letter[l] = i
mapping_letter = {v:k for k,v in mapping_letter.items()}


# Display some pictures of the dataset
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(8, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    img = training_images[i].reshape(28,28)
    ax.imshow(img, cmap = 'gray')
    title = mapping_letter[training_labels[i]]
    ax.set_title(title, fontsize = 15)
plt.tight_layout(pad=0.5)
#plt.show()

# Display the distribution of each letter

vc = pd.Series(training_labels).value_counts()
plt.figure(figsize=(20,5))
sns.barplot(x = sorted(vc.index), y = vc, palette = "rocket")
plt.title("Number of pictures of each category", fontsize = 15)
plt.xticks(fontsize = 15)
#plt.show()

training_images = np.expand_dims(training_images, axis = 3)
testing_images = np.expand_dims(testing_images, axis = 3)

print(training_images.shape)
print(testing_images.shape)

# Create an ImageDataGenerator and do Image Augmentation

train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                                   height_shift_range=0.1,
                                   width_shift_range=0.1,
                                   zoom_range=0.1,
                                   shear_range=0.1,
                                   rotation_range=10,
                                   fill_mode='nearest',
                                   horizontal_flip=True)

#Image Augmentation is not done on the testing data

validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_datagenerator = train_datagen.flow(training_images,
                                         training_labels,
                                         batch_size = 32)

validation_datagenerator = validation_datagen.flow(testing_images,
                                                   testing_labels, 
                                                   batch_size=32)
                                                   
# Define a Callback class that stops training once accuracy reaches 99.8%

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')is not None and logs.get('accuracy')>=0.998):
      print("\nReached 99.8% accuracy so cancelling training!")
      self.model.stop_training = True   


# Define the model

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape = (28,28,1)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(25, activation = 'softmax')])

model.summary()

# Compiling the Model. 
model.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])
              
              
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience = 2, 
                                            verbose=1,factor=0.25, 
                                            min_lr=0.0001)          


# Train the Model
callbacks = myCallback()
history = model.fit(train_datagenerator,
                    validation_data = validation_datagenerator,
                    steps_per_epoch = len(training_labels)//32,
                    epochs = 12,
                    validation_steps = len(testing_labels)//32,
                    callbacks = [callbacks, learning_rate_reduction])

# Plot the chart for accuracy and loss on both training and validation

model.save('sign_language.h5')

model.evaluate(testing_images, testing_labels, verbose=0)




      