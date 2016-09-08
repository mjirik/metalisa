from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py

def nacteni(soubor):
    a = list()
    b = list()
    c = h5py.File(soubor, 'r')
    pocet = 0
    for i in c.keys():
        gr = c[i]
        for j in gr.keys():
            dts = gr[j]
            atr = dts.attrs
            #x = np.zeros(3)
            #x[atr['teacher'] - 1] = 1
            #a.append(x)
	    if atr['teacher'] == 2:
	    	a.append(1)
	    else:
	    	a.append(0)
            b.append(dts[:,:])
	    pocet +=1
            #b.append(abs(dts[:, :]) / float(np.max(abs(dts[:, :]))))
    a = np.asarray(a)
    b = np.asarray(b)
    b = b.reshape(pocet, 1, 40, 40)
    return a, b


def main():
    testl, testd = nacteni('test.hdf5')
    trainl, traind = nacteni('train.hdf5')
    print testl.shape , testd.shape
    vall = testl[:700]
    vald = testd[:700,:,:,:]
    testl = testl[700:]
    testd = testd[700:,:,:,:]
    seed = 7
    np.random.seed(seed)

    # datagen = ImageDataGenerator(
    #     rotation_range=40,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     fill_mode='nearest')

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, 40, 40)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    #model.add(Dense(3))
    model.add(Dense(1))
    #model.add(Activation('softmax'))
    model.add(Activation('sigmoid'))
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(traind, trainl, batch_size=30, nb_epoch=50, validation_data=(vald,vall), shuffle=True)

    scores = model.evaluate(testd, testl, batch_size = 32)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    ##ulozeni natrenovaneho modelu
    # model_json = model.to_json()
    # with open('model.json', 'w') as json_file:
    #     json_file.write(model_json)
    model.save_weights('vahy1.hdf5')
    with open('vysledek1.txt', 'w') as f:
        f.write("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	f.write(history.history)
    # print('Saved model to disk')
if __name__ == "__main__":
    main()
