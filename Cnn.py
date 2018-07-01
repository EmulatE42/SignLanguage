import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils
from keras import backend as K

size = 100


def makeXY(file):
    df = pd.read_csv(file)
    prvi = df["label"]
    SVI = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    p = []
    d = []
    for i in prvi:
        p.append(SVI.index(i))
        print("DODAJEM", SVI.index(i))
    p = np.array(p)

    for i in range(1, size * size + 1):
        L = []
        d.append(np.array(df["pixel" + str(i)].values))
    d = np.array(d).transpose(1, 0)
    aa = []
    for i in d:
        aa.append(i.reshape(size, size))
    aa = np.array(aa)
    return aa, p


K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load data
X_train, y_train = makeXY("train.csv")
X_test, y_test = makeXY("test.csv")

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, size, size).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 1, size, size).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define the larger model
def larger_model():
    # create model
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=(1, size, size)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = larger_model()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))


# model_json = model.to_json()
# with open("SIGN.json", "w") as json_file:
#    json_file.write(model_json)
# model.save_weights("SIGN.h5")
