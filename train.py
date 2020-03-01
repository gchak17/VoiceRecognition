import warnings
warnings.filterwarnings("ignore")
from keras import optimizers
from keras.models import Sequential, load_model
from numpy import asanyarray, save, load, argmax
from keras.layers import Dense, Activation, Flatten, Conv2D, AveragePooling2D, Dropout, MaxPooling2D
from random import randint
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
from keras.callbacks.callbacks import ModelCheckpoint
    
def cnn_model(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.40))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.45))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))
    model.add(Dense(classes, activation='softmax'))
    adam = optimizers.rmsprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def main():
    train_x = load('train_x.npy')
    train_y = load('train_y.npy')

    train_x /= 255
    model = cnn_model((train_x.shape[1], train_x.shape[2], train_x.shape[3]), train_y.shape[1])
    model.fit(train_x, train_y, epochs = 150, batch_size = 32, verbose=1)
    model.save('model.h5')

    #plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    #plt.legend(['training', 'validation'], loc = 'upper left')
    #plt.show()
    pass

if __name__ == "__main__":
    main()