# dataset : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/code
# chest Xray dataları ile modeli eğitme ve tahmin. acx= 84.99
from statistics import mode
from xml.parsers.expat import model
import numpy as np 
from keras.models import Sequential
from tensorflow.keras .layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt



#Image Preprocessing
train_datagen = ImageDataGenerator(rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255) 

training_set = train_datagen.flow_from_directory('chest_xray/train',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('chest_xray/val/',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory('chest_xray/test',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')


#CNN Model
model = Sequential()

model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(64, 64, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))


model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()


#modelimizi fit eedelim
history = model.fit(training_set,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps = 25, steps_per_epoch=25)
    

test_acc = model.evaluate(test_set,steps=624)
print('*'*100,'\n\tThe testing accuracy is :',test_acc[1]*100, '%')



#prediction
Y_pred = model.predict(test_set, 100)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)

"""plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()"""