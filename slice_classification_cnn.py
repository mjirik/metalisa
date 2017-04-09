# -*- coding: utf-8 -*-
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py
import argparse
import os.path as op
import cPickle as pickle
import scipy.ndimage.interpolation as interpolation
import random


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


def nacteni_c(soubor):
    a = list()
    b = list()
    c = h5py.File(soubor, 'r')
    pocet = 0
    for i in c.keys():
        gr = c[i]
        for j in gr.keys():
            if j == '0':
                continue
            name = int(j)
            prev = gr[str(name-1)]
            try:
                next = gr[str(name+1)]
            except KeyError:
                continue

            dts = gr[j]

            atr = dts.attrs
            x = np.zeros(3)  #
            x[atr['teacher'] - 1] = 1  # 3 třídy
            a.append(x)  #

            pocet += 1

            res = np.concatenate((prev[:,:],dts[:,:],next[:,:] ),axis=1)
            b.append(abs(res[:, :]) / float(np.max(abs(res[:, :]))))
            # for i in range(5, 11, 5):
            #     pocet += 1
            #     prevR = interpolation.rotate(prev, i, reshape=False, cval=-1024)
            #     dtsR = interpolation.rotate(prev, i, reshape=False, cval=-1024)
            #     nextR = interpolation.rotate(prev, i, reshape=False, cval=-1024)
            #     res = np.concatenate((prevR[:, :], dtsR[:, :], nextR[:, :]), axis=1)
            #     b.append(abs(res[:, :]) / float(np.max(abs(res[:, :]))))
            #     a.append(x)
            #
            #     pocet += 1
            #     prevR = interpolation.rotate(prev, -i, reshape=False, cval=-1024)
            #     dtsR = interpolation.rotate(prev, -i, reshape=False, cval=-1024)
            #     nextR = interpolation.rotate(prev, -i, reshape=False, cval=-1024)
            #     res = np.concatenate((prevR[:, :], dtsR[:, :], nextR[:, :]), axis=1)
            #     b.append(abs(res[:, :]) / float(np.max(abs(res[:, :]))))
            #     a.append(x)

    a = np.asarray(a)
    b = np.asarray(b)

    b = b.reshape(pocet, 1, b.shape[1], b.shape[2])
    return a, b


def test_picture_c(soubor):
    a = list()
    b = list()
    c = h5py.File(soubor, 'r')
    pocet = 0
    for i in c.keys():
        gr = c[i]
        ax = len(gr.keys())
        for j in range(ax-1):
            if j==0:
                dts = gr[str(j)]
                continue
            prev = dts
            next = gr[str(j+1)]
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
            # b.append(abs(dts[:, :]) / float(np.max(abs(dts[:, :]))))
            res = np.concatenate((prev[:, :], dts[:, :], next[:, :]), axis=1)
            b.append(abs(res[:, :]) / float(np.max(abs(res[:, :]))))
            origin = atr['origin file']
        break
    a = np.asarray(a)
    b = np.asarray(b)
    b = b.reshape(pocet, 1, b.shape[1], b.shape[2])
    return b, a, origin


def linearization(data, margin_sz=4):
    data = np.squeeze(np.asarray(data))
    conv_str = np.asarray([1, 2, 3, 2, 1]) / 9.
    conv_data = np.zeros(data.shape)
    # print data.shape
    conv_data[:, 0] = np.convolve(data[:, 0], conv_str, 'same')
    conv_data[:, 1] = np.convolve(data[:, 1], conv_str, 'same')
    conv_data[:, 2] = np.convolve(data[:, 2], conv_str, 'same')

    # beginning

    if len(data) > margin_sz:
        conv_start = np.asarray([3, 2, 1]) / 6.0
        conv_end = np.asarray([1, 2, 3]) / 6.0
        valid_sz = margin_sz - len(conv_start) + 1

        conv_data[:valid_sz, 0] = np.convolve(data[:margin_sz, 0], conv_start, 'valid')
        conv_data[:valid_sz, 1] = np.convolve(data[:margin_sz, 1], conv_start, 'valid')
        conv_data[:valid_sz, 2] = np.convolve(data[:margin_sz, 2], conv_start, 'valid')

        conv_data[-valid_sz:, 0] = np.convolve(data[-margin_sz:, 0], conv_end, 'valid')
        conv_data[-valid_sz:, 1] = np.convolve(data[-margin_sz:, 1], conv_end, 'valid')
        conv_data[-valid_sz:, 2] = np.convolve(data[-margin_sz:, 2], conv_end, 'valid')

    print conv_data


def generator(labels, images):
    while 1:
        rnd = random.sample(zip(labels, images), 30)
        aug_im = list()
        l = list()
        for i in np.squeeze(rnd):
            image = np.squeeze(i[1])
            label = i[0]
            a = image.shape
            im1 = np.array(image[:,:a[1]/3])
            im2 = np.array(image[:,a[1]/3:2*a[1]/3])
            im3 = np.array(image[:,2*a[1]/3:])
            for ang in range(-15,16):
                prevR = interpolation.rotate(im1, ang, reshape=False, cval=-1024)
                dtsR = interpolation.rotate(im2, ang, reshape=False,cval=-1024)
                nextR = interpolation.rotate(im3, ang, reshape=False,cval=-1024)
                aug_im.append(np.concatenate((prevR[:, :], dtsR[:, :], nextR[:, :]), axis=1))
                l.append(label)
        a = np.asarray(aug_im)
        s = a.shape
        aug_im = a.reshape(s[0], 1, s[1], s[2])
        yield np.asarray(aug_im), np.asarray(l)


def main():
    parser = argparse.ArgumentParser(description='CNN training')
    parser.add_argument('-i', '--input-data-dir', type=str,
                        help='input data')
    parser.add_argument('-o', '--output-data-dir', type=str,
                        help='output data')
    parser.add_argument('-t', '--test-data-dir', type=str,
                        help='whole experiment')
    parser.add_argument('-m', '--model', type=str, help='model file')
    parser.add_argument('-c', '--context', action='store_true', help='use context of image')
    parser.add_argument('-a', '--augmentation', action='store_true', help='augmentation use')
    args = parser.parse_args()


    args.input_data_dir = op.expanduser(args.input_data_dir)
    if args.output_data_dir:
        args.output_data_dir = op.expanduser(args.output_data_dir)
    args.test_data_dir = op.expanduser(args.test_data_dir)

    testfile = op.join(args.input_data_dir, 'test.hdf5')
    trainfile = op.join(args.input_data_dir, 'train.hdf5')
    if args.output_data_dir:
        outputfile = op.join(args.output_data_dir, 'vahy.hdf5')
    if args.context:
        testl, testd = nacteni_c(testfile)


        trainl, traind = nacteni_c(trainfile)

        td, tl, orig = test_picture_c(testfile)
    else:
        testl, testd = nacteni(testfile)


        trainl, traind = nacteni(trainfile)

        td, tl, orig = test_picture(testfile)

    res = testd.shape[2:]
    seed = 7
    np.random.seed(seed)
    if args.augmentation:
        datagen = ImageDataGenerator(
            rotation_range=40,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # shear_range=0.2,
            # zoom_range=0.2,
            # horizontal_flip=True,
            fill_mode='nearest')

        gen = datagen.flow(traind, trainl, batch_size=30)

    if args.context & args.augmentation:
        myGen = generator(trainl, traind)

    if args.model:
        json_file = open(args.model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model = Sequential()
        model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(1, res[0], res[1])))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(16, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(16, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(3))
        model.add(Activation('softmax'))  # 3tridy
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # model.add(Dense(1))
        # model.add(Activation('sigmoid')) # 2tridy
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print 'trenovani'
    if (args.augmentation) & (not args.context):
        model.fit_generator(gen, samples_per_epoch=30, nb_epoch=500)

    elif args.augmentation & args.context:
        model.fit_generator(myGen, samples_per_epoch=930, nb_epoch=150)
    else:
        model.fit(td, tl, batch_size=32, nb_epoch=100)

    scores = model.evaluate(testd, testl)
    vis = list()
    vis.append(orig)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    for ind, i in enumerate(td):
        sh = i.shape
        i = i.reshape(1, 1, sh[1], sh[2])
        aaa = model.predict(i)
        vis.append(aaa)
        print aaa, tl[ind]


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
