# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: ds_env
#     language: python
#     name: ds_env
# ---

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow import keras

import os
from bs4 import BeautifulSoup

# %% [markdown]
# ## Loading and preprocessing Data

# %%
img_folder = './images/'
annot_folder = './annotations/'

# %% [markdown]
# ### Extracting image name and class from xml file

# %%
desc = []
for dirname, _, filenames in os.walk(annot_folder):
    for filename in filenames:
        desc.append(os.path.join(dirname, filename));

# %%
img_name,label = [],[]

for d in desc:
    content = []
    n = []

    with open(d, "r") as file:
        content = file.readlines()
    content = "".join(content)
    soup = BeautifulSoup(content,"html.parser")
    file_name = soup.filename.string
    name_tags = soup.find_all("name")
    

    for t in name_tags:
        n.append(t.get_text())
        
    # slecting tag with maximum occurence in an image (If it has multiple tags)
    name = max(set(n), key = n.count)
  
    img_name.append(file_name)
    label.append(name)
 

# %%
# One Hot Encoding label data
labels = pd.get_dummies(label)
labels.head()

# %%
# Our target classes
classes = list(labels.columns)
classes

# %% [markdown]
# ### Loading Images and converting them to pixel array

# %%
data, target = [],[]
img_h, img_w = 256, 256

for i in range(len(img_name)):
    name = img_name[i]
    path = img_folder + name
    
    image = load_img(path, target_size = (img_h, img_w))
    image = img_to_array(image)
    data.append(image)
    target.append(tuple(labels.iloc[i,:]))

# %%
# Convering list to array
data=np.array(data,dtype="float32")/255.0
target=np.array(target,dtype="float32")

# %% [markdown]
# ### Visualizing few images randomly

# %%
plt.figure(figsize=(10, 10))
for i,j in enumerate(np.random.randint(1, 500, 9, dtype=int)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(data[j])
    plt.title(label[j])
    plt.axis("off")

# %%
data.shape, target.shape

# %%
# Splitting into train and test data
train_img,test_img,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=20)

# %%
print("Train shapes : ",(train_img.shape, y_train.shape))
print("Test shapes : ",(test_img.shape, y_test.shape))

# %% [markdown]
# ### Generating and fitting Model

# %%
data_augmentation = Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_h, img_w, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1)
  ]
)

# %%
checkpoint_cb = keras.callbacks.ModelCheckpoint("face_model.h5",
                                                    save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
                                                      restore_best_weights=True)

# %%
num_classes = 3
model = Sequential([
    data_augmentation,
    layers.Conv2D(16, 3,padding = "same",input_shape=(img_h, img_w, 3)) ,
    layers.LeakyReLU(),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3,padding = "same"),
    layers.LeakyReLU(),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3,padding = "same"),
    layers.LeakyReLU(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128),
    layers.LeakyReLU(),
    layers.Dense(num_classes,activation = "softmax")
])

# %%
model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'],
              )

# %% _kg_hide-output=true
epochs = 1000
history = model.fit(train_img,y_train,
                    validation_data=(test_img,y_test),
                    batch_size=32,
                    epochs=epochs,
                    callbacks=[checkpoint_cb, early_stopping_cb])

# %% [markdown]
# * Final validation loss = 0.5871
# * Final validation accuracy = 0.7778

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = history.epoch

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %% [markdown]
# ## Checking predictions

# %%
model_to_predict = keras.models.load_model('face_model.h5')

# %%
y_pred_1 = model_to_predict.predict(test_img[100].reshape(-1,img_h,img_w,3))
print(classes[np.argmax(y_pred_1)])
plt.imshow(test_img[100])

# %%
y_pred_2 = model.predict(test_img[3].reshape(-1,img_h,img_w,3))
print(classes[np.argmax(y_pred_2)])
plt.imshow(test_img[3])

# %%
