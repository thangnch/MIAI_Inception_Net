from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

CLASS_NUM = 196
BATCH_SIZE = 32
EPOCHS = 100
IMG_SIZE = (256,256, 3)

incept_model = InceptionV3(input_shape=IMG_SIZE, include_top = False, weights = "imagenet")

for layer in incept_model.layers:
    layer.trainable = False

x = Flatten()(incept_model.output)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(CLASS_NUM, activation="softmax")(x)

model = Model(incept_model.input, x)
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['acc'])

model.summary()

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = "data/train"
test_dir = "data/test"

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=BATCH_SIZE, class_mode="categorical", target_size=IMG_SIZE[:-1])
test_generator = test_datagen.flow_from_directory(test_dir, batch_size=BATCH_SIZE, class_mode="categorical", target_size=IMG_SIZE[:-1])

print(train_generator.class_indices)

model.fit_generator(train_generator, validation_data=test_generator, steps_per_epoch=train_generator.n//BATCH_SIZE, epochs = EPOCHS,
                    validation_steps=test_generator.n//BATCH_SIZE,verbose=1 )
model.save("mymodel.h5")


