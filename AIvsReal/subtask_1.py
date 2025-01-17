import tensorflow as tf
from tensorflow.keras.utils import to_categorical, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import os
import pandas as pd
import numpy as np
from PIL import Image
pd.DataFrame().to_csv(r'/content/induction-task/Data/result.csv',index=False,header=False)
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)
strategy = tf.distribute.MirroredStrategy()
tf.random.set_seed(1234)
def features(images):
    x_features=[]
    for image in images:
        try:
            x_features.append((np.array(tf.keras.utils.load_img(image,target_size=(299,299)))))
        except Exception as e:
            print(f"Error loading image {image}: {e}")
    return np.array(x_features)
def create_data(dir):
    paths = []
    labels = []
    for label in os.listdir(dir):
        for image in os.listdir(os.path.join(dir, label)):
          try:
            tf.keras.utils.load_img(os.path.join(dir, label, image))
            paths.append(os.path.join(dir, label, image))
            labels.append(label)
          except Exception as e:
            print(f"Error loading image {os.path.join(dir, label, image)}: {e}")
    data = pd.DataFrame()
    data['Paths'], data['labels'] = paths, labels
    return data

data = create_data(r"/content/induction-task/Data/Train")
s_data = data.sample(frac=1)
x_train = features(s_data['Paths'])
y_train = [0 if label == 'AI' else 1 for label in s_data['labels']]
y_train = np.array(y_train).reshape((-1, 1))
base = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3), pooling='max')
base.trainable = False

model = Sequential([
    base,
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy', 'precision', 'Recall'])

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=[0.8,1.2],
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest',
)
checkpoint = ModelCheckpoint(
    r'/content/induction-task/Data/mymodel.keras',  
    monitor='val_loss',     
    verbose=1,             
    save_best_only=True,     
    mode='min',         
    save_weights_only=False, 
    save_freq='epoch'        
)

model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=164,
          callbacks=[early_stopping, reduce_lr,checkpoint])
base.tarinable=True
for layer in base.layers[-30:]:
    layer.tarinable=True
    layer.kernel_regularizer=regularizers.l2(0.01)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy', 'precision', 'Recall'])
model.fit(datagen.flow(x_train, y_train, batch_size=16), epochs=15,
          callbacks=[early_stopping, reduce_lr])

result = pd.DataFrame()
id = []
pred = []
path_t = []

for path in os.listdir(r'/content/induction-task/Data/Test'):
    try:
       if os.path.join(r'/content/induction-task/Data/Test', path) != r'/content/induction-task/Data/Test/image_62.jpg':
            path_t.append(os.path.join(r'/content/induction-task/Data/Test', path))
            id.append(path.split('.')[0])
    except Exception as e:
        print(f"Error loading image {path}: {e}")

x_test = preprocess_input(features(path_t))
y_res = model.predict(x_test, verbose=1, batch_size=16).flatten()

for i in y_res:
    if i > 0.5:
        pred.append('Real')
    else:
        pred.append('AI')

result['Id'] = id
result['Label'] = pred
result.to_csv(r'/content/induction-task/Data/result.csv',index=False)
pd.set_option('display.max_rows', None) 
print(result)
model.save(r'/content/induction-task/Data/mymodel.keras')