# -*- coding: utf-8 -*-
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py
import argparse
import os.path as op
import cPickle as pickle


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
            x = np.zeros(3)  #
            x[atr['teacher'] - 1] = 1  # 3 třídy
            a.append(x)  #

            # if atr['teacher'] == 2:     #
            #   a.append(1)              #
            # else:                       # 2 třídy
            #   a.append(0)              #

            # b.append(dts[:,:])   #bez prumerovani

            pocet += 1
            b.append(abs(dts[:, :]) / float(np.max(abs(dts[:, :]))))  # s prumerovanim

    a = np.asarray(a)
    b = np.asarray(b)

    b = b.reshape(pocet, 1, b.shape[1], b.shape[2])
    return a, b

def test_picture(soubor):
    a = list()
    b = list()
    c = h5py.File(soubor, 'r')
    pocet = 0
    for i in c.keys():
        gr = c[i]
        ax = len(gr.keys())
        for j in range(ax):
            dts = gr[str(j)]
            atr = dts.attrs
            x = np.zeros(3)
            x[atr['teacher'] - 1] = 1
            a.append(x)
            # if atr['teacher'] == 2:
            #   a.append(1)
            # else:
            #   a.append(0)
            # b.append(dts[:,:])
            pocet += 1
            b.append(abs(dts[:, :]) / float(np.max(abs(dts[:, :]))))
            origin = atr['origin file']
        break
    a = np.asarray(a)
    b = np.asarray(b)
    b = b.reshape(pocet, 1, b.shape[1], b.shape[2])
    return b, a, origin


def main():
    parser = argparse.ArgumentParser(description='CNN training')
    parser.add_argument('-i', '--input-data-dir', type=str,
                        help='input data')
    parser.add_argument('-o', '--output-data-dir', type=str,
                        help='output data')
    parser.add_argument('-t', '--test-data-dir', type=str,
                        help='whole experiment')
    parser.add_argument('-m', '--model', type=str, help='model file')
    args = parser.parse_args()


    args.input_data_dir = op.expanduser(args.input_data_dir)
    if args.output_data_dir:
        args.output_data_dir = op.expanduser(args.output_data_dir)
    args.test_data_dir = op.expanduser(args.test_data_dir)

    testfile = op.join(args.input_data_dir, 'test.hdf5')
    trainfile = op.join(args.input_data_dir, 'train.hdf5')
    if args.output_data_dir:
        outputfile = op.join(args.output_data_dir, 'vahy.hdf5')
    testl, testd = nacteni(testfile)
    res = testd.shape[2]

    trainl, traind = nacteni(trainfile)

    td, tl, orig = test_picture(testfile)

    seed = 7
    np.random.seed(seed)

    datagen = ImageDataGenerator(
        rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        fill_mode='nearest')

    gen = datagen.flow(traind, trainl, batch_size=30)
    if args.model:
        json_file = open(args.model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model = Sequential()
        model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(1, res, res)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(1, res, res)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(3))
        model.add(Activation('softmax'))  # 3tridy
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # model.add(Dense(1))
        # model.add(Activation('sigmoid')) # 2tridy
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit_generator(gen, samples_per_epoch=30, nb_epoch=500)

    scores = model.evaluate(testd, testl)
    vis = list()
    vis.append(orig)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    for ind, i in enumerate(td):
        i = i.reshape(1, 1, res, res)
        aaa = model.predict(i)
        vis.append(aaa)
        print np.around(aaa), tl[ind]


    if args.test_data_dir:
        model_json = model.to_json()
        with open( args.test_data_dir + '/model.json', 'w') as json_file:
            json_file.write(model_json)
            model.save_weights(args.test_data_dir + '/vahy.hdf5')
        with open(args.test_data_dir + '/parametry.txt', 'w') as f:
            f.write('pocet trid:' + str(len(testl[1])+1) + '\n')
            f.write('vstupni data:' + (args.input_data_dir))
        with open(args.test_data_dir + '/vis.pkl','w' ) as f:
            pickle.dump(vis,f)





    print('Saved model to disk')
if __name__ == "__main__":
    main()
