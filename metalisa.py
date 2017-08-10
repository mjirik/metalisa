#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © %YEAR%  <>
#
# Distributed under terms of the %LICENSE% license.

"""

"""

import logging
logger = logging.getLogger(__name__)
import argparse
import glob
import numpy as np
import io3d
import pandas as pd
import os.path
import io3d.datareader as DR
import yaml
import datetime
import time
import scipy.ndimage.interpolation as interpolation
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras import optimizers

def stats(config_file, stat_path, labels, args, results):
    args = vars(args)
    config = config_parse(config_file)
    names = config.keys() + args.keys() + ['labels']  + ['tr_result'] #+ ['date']
    data = config.values() + args.values() + [labels]  + [results] #+ [time.asctime(datetime.datetime.now())]
    print len(names)
    print len(data)
    #names.append('labels').append('date').append('tr_result')
    #data.append(labels).append(time.asctime(datetime.datetime.now())).append(results)


    if os.path.exists(stat_path):
        df0 = pd.read_csv(stat_path)
    else:
        df0 = pd.DataFrame()
    new_df = df0
    dt = dict(zip(names, data))
    df = pd.DataFrame(dt)
    new_df = pd.concat([df, new_df])
    new_df.to_csv(stat_path, index=False)
    return


def config_parse(config_file='config.txt'):
    config = open(config_file,'r')
    con = dict()
    for i in config:
        a = i.split('=')
        con[a[0]] = a[1].strip()

    return con


def model_creation(config_file, shape, context):
    con = config_parse(config_file)

    model = Sequential()

    for i in range(int(con['layers'])):
        for j in range(int(con['sub_layers'])):
            if i == 0 & j ==0:
                if context:
                    model.add(Convolution2D(int(con['filters']), int(con['filter_size']), int(con['filter_size']), border_mode='same', input_shape=(1, 3*shape[1], shape[2])))
                else:
                    model.add(Convolution2D(int(con['filters']), int(con['filter_size']), int(con['filter_size']), border_mode='same', input_shape=(1, shape[1], shape[2])))
            else:
                model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(Activation(con['activation']))
        model.add(MaxPooling2D(pool_size=(int(con['max_polling_size']), int(con['max_polling_size']))))
    model.add(Flatten())
    model.add(Dense(int(con['dense_filters'])))
    model.add(Activation(con['activation']))
    model.add(Dropout(float(con['dropout'])))

    model.add(Dense(int(con['output_neurons'])))
    if int(con['output_neurons']) < 3:
        model.add(Activation('sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Activation('softmax'))
        sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_data(filename, csv=None , axis=0):
    data3d, _ = DR.read(filename)
    labels = []
    slab = []
    if csv:
        df = pd.read_csv(csv)
        labs = np.unique(df["label"])
        vals = range(len(labs))

        slab = dict(zip(vals, labs))
        rev_slab = dict(zip(labs, vals))

        fn_unique = np.unique(df["filename"])

        name = filename.split('/')[-1]

        file_list = [i for i, item in enumerate(fn_unique) if item.endswith(name)]


        index = file_list[0]

        this_fn = fn_unique[index]
        one_file = df[df["filename"] == this_fn]

        maximum_slice_number = np.max(np.max(one_file[["start_slice_number", "stop_slice_number"]]))

        labels = [None] * maximum_slice_number

        dct = one_file[["label", "start_slice_number", "stop_slice_number"]].to_dict()
        start = dct['start_slice_number'].values()
        stop = dct['stop_slice_number'].values()
        label = dct['label'].values()
        for i in vals:
            a = start[i]
            b = stop[i]
            lab = rev_slab[label[i]]
            c = [lab] * (b + 1 - a)
            labels[a:b + 1] = c
        for ind, position in enumerate(labels):
            zer = np.zeros(len(vals))
            zer[position - 1] = 1
            labels[ind] = zer

    if len(data3d.shape) == 3:
        if axis != 0:
            data3d = np.rollaxis(data3d, axis)
        images = list()
        for i in data3d:
            images.append(i)

    else:
        if axis != 0:
            print 'jednotlivé řezy musí být otočeny kolem správné osy'
        images = data3d



    return images, labels, slab


def train(trainpath, modelpath,  context=False, augmentation=False, csvpath=None, config_file='config.txt', axis=0):

    if csvpath is None:
        csvfiles = glob.glob(os.path.join(trainpath, '*.csv'))
        csvpath = csvfiles[0]
        if len(csvfiles) > 1:
            logger.error("There is more .csv files in train directory. Please select one with csvpath parameter.")
    tl = list()
    td = list()
    image_end = list()
    for filename in glob.glob(os.path.join(trainpath, '*.pklz')):
        logger.info(filename)
        image, labels, slab = load_data(filename, csvpath)
        td += image
        tl += labels
        image_end.append(len(td))

    td = np.array(td)
    shape = td.shape
    td = td.reshape(shape[0], 1, shape[1], shape[2])
    tl = np.array(tl)

    try:
        model = model_creation(config_file, shape, context)
    except:
        print 'defaultni model'
        model = Sequential()
        if context:
            model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, 3*shape[1], shape[2])))
        else:
            model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, shape[1], shape[2])))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(3))
        model.add(Activation('softmax'))  # 3tridy
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if augmentation or context:
        history = model.fit_generator(train_generator(td, tl, image_end, augmentation, context),32, 100)
    else:
        history = model.fit(td, tl, batch_size=32, nb_epoch=10)

    with open(modelpath, 'w') as json_file:
        model_json = model.to_json()
        json_file.write(model_json)

    with open(modelpath[:-5]+'.yml', 'w') as f:
        yaml.dump(slab, f)
    return history, slab

def validation(predictpath, modelpath, csvpath=None, context=False, axis=0):

    json_file = open(modelpath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    if csvpath is None:
        csvfiles = glob.glob(os.path.join(predictpath, '*.csv'))
        csvpath = csvfiles[0]
    tl = list()
    td = list()
    image_end = list()
    for filename in glob.glob(os.path.join(predictpath, '*.pklz')):
        logger.info(filename)
        image, labels, slab = load_data(filename, csvpath)
        td += image
        tl += labels
        image_end.append(len(td))

    td = np.array(td)
    shape = td.shape
    td = td.reshape(shape[0], 1, shape[1], shape[2])
    tl = np.array(tl)

    if context:
        history = model.evaluate_generator(generator_context_eval(td, tl, image_end), 1000, 50)
    else:
        history = model.evaluate(td, tl, batch_size=32)

    return history


def predict(predictpath, modelpath, csvpath="prediction.csv", context=False, axis=0):

    json_file = open(modelpath, 'r')
    slab = yaml.load(modelpath[:-5]+'.yml')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)


    sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    vysledky = list()
    names = list()
    slice = list()
    text_result = list()
    validation_label = list()
    for filename in glob.glob(os.path.join(predictpath, '*.pklz')):
        logger.info(filename)
        image, _, _ = load_data(filename)
        num = 0
        for i in image:
            if context:
                a = i.reshape(1, 1, i.shape[0], i.shape[1])
                vysledky.append(model.predict_generator(generator_context_predict(a),1,1))
            else:
                a = i.reshape(1, 1, i.shape[0], i.shape[1])
                vysledky.append(model.predict(a))
            names.append(filename)
            slice.append(num)
            num += 1
            text_result.append(slab[np.argmin(vysledky[-1])])


    if os.path.exists(csvpath):
        df0 = pd.read_csv(csvpath)
    else:
        df0 = pd.DataFrame()
    new_df = df0
    dt = {'Filename': names, 'Slice_number': slice, 'Numeric_Label': vysledky,  'Text_Label': text_result}
    df = pd.DataFrame(dt)
    new_df = pd.concat([df,new_df])
    new_df.to_csv(csvpath, index=False)



    return new_df, vysledky


def bounding_box(train_path, predict_path, model_path, augmentation, context, config_path):
    if train_path & predict_path:
        for i in range(3):
            model_path_part =  model_path.split('.')
            if len(model_path_part) > 2:
                pass
            model_path = model_path_part[-2]+ '_'+ str(i)+ '.' + model_path_part[-1]
            train(train_path, model_path, augmentation, context, config_file=config_path, axis=i)
    pass


def sample_data(trainpath):
    '''vytvoření sample dat ulozeni do cesty a vraci je pro dalsi praci'''
    filename = os.path.join(trainpath, "data01.tiff")
    labeling_path = os.path.join(trainpath, "train.csv")

    data3d, labeling = sample_one_data()
    save_one_sample_data(filename, labeling_path, data3d, labeling)

    return data3d

def save_one_sample_data(filename, labeling_path, data3d, labeling):
    io3d.datawriter.write(data3d, filename, metadata={"voxelsize_mm": [1, 1, 1]})
    labeling["filename"] = [filename, filename, filename]

    df = pd.DataFrame(labeling)

    if os.path.exists(labeling_path):
        df0 = pd.read_csv(labeling_path)
    else:
        df0 = pd.DataFrame()
    new_df = pd.concat([df0, df])

    new_df.to_csv(labeling_path)



def sample_one_data():

    datasize = [300, 512, 512]

    data3d = np.ones(shape=datasize, dtype=np.int16) * -1024
    heath_intensity = 230
    liver_intensity = 130

    start_slice_number = 55
    stop_slice_number = 100
    data3d[:, 40:-70, 45:-30] = 100
    data3d[:, 50:-80, 55:-40] = -100
    data3d[0:50, 45:-80, 55:-40] = -800
    data3d[20:55, 310:370, 200:300] = heath_intensity
    data3d[start_slice_number:stop_slice_number, 200:400, 60:400] = liver_intensity
    noise = (np.random.rand(*datasize) * 100).astype(np.int16)
    data3d += noise

    metadata = {
        "label": ["under_liver", "liver", "above_liver"],
        "start_slice_number": [0, start_slice_number, stop_slice_number],
        "stop_slice_number": [start_slice_number - 1, stop_slice_number - 1, datasize[0]],
        "filename": [None, None, None]
    }


    return data3d, metadata

def train_generator(images, labels, ends, augmentation=False ,context=False):
    while 1:
        ind = np.random.random_integers(0, len(labels)-1)
        img = images[ind]
        angle = np.random.random_integers(-20, 20)
        if context:
            if ind == 0:
                im1 = img
            if ind not in ends:
                im1 = images[ind-1]
            else:
                im1 = img
            im2 = img
            if (ind + 1) not in ends:
                im3 = images[ind+1]
            else:
                im3 = img
            if augmentation:
                im1 = interpolation.rotate(im1, angle, reshape=False, cval=-1024)
                im2 = interpolation.rotate(im2, angle, reshape=False, cval=-1024)
                im3 = interpolation.rotate(im3, angle, reshape=False, cval=-1024)
            img = np.concatenate((im1[:, :], im2[:, :], im3[:, :]), axis=1)
        else:
            img = interpolation.rotate(img, angle, reshape=False, cval=-1024)
        label = labels[ind]
        a = np.asarray(img)
        s = a.shape
        aug_im = a.reshape(1, 1, s[1], s[2])
        yield np.asarray(aug_im), label.reshape(1, label.shape[0])


def generator_context_eval(images, labels, ends):
    while 1:
        for ind in range(len(images)):
            img = images[ind]
            if ind == 0:
                im1 = img
            if ind not in ends:
                im1 = images[ind - 1]
            else:
                im1 = img
            im2 = img
            if (ind + 1) not in ends:
                im3 = images[ind + 1]
            else:
                im3 = img
            print im1.shape, im2.shape, im3.shape
            img = np.concatenate((im1[:, :], im2[:, :], im3[:, :]), axis=1)
            label = labels[ind]
            a = np.asarray(img)
            s = a.shape
            aug_im = a.reshape(1, 1, s[1], s[2])
            yield np.asarray(aug_im), label.reshape(1, label.shape[0])


def generator_context_predict(images):
    while 1:
        for ind in range(len(images)):
            img = images[ind]
            if ind == 0:
                im1 = img
            else:
                im1 = img[ind-1]
            im2 = img
            if (ind+1) == len(images):
                im3 = images[-1]
            else:
                im3 = img

            img = np.concatenate((im1[:, :], im2[:, :], im3[:, :]), axis=1)
            a = np.asarray(img)
            s = a.shape
            print s
            aug_im = a.reshape(s[0], 1, s[1], s[2])
            yield np.asarray(aug_im)# , label


def main():
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    # fh = logging.FileHandler('log.txt')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.debug('start')

    # input parser
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    # parser.add_argument(
    #     '-p', '--parameterfile',
    #     default=None,
    #     # required=True,
    #     help='input parameter file'
    # )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')

    parser.add_argument(
        '-csd', '--create-sample-data', action='store_true',
        help='Debug mode')

    parser.add_argument(
        '-l', '--logfile',
        default="~/teigen.log",
        help='Debug mode')

    parser.add_argument(
        '-t', '--train', action='store_true',
        help='Run train')

    parser.add_argument(
        '-tp', '--trainpath',
        default="train",
        help='Train path')

    parser.add_argument(
        '-conf', '--configpath',
        default='config.txt',
        help='config file path')

    parser.add_argument(
        '-p', '--predict', action='store_true',
        help='Run prediction')
    parser.add_argument(
        '-pp', '--predictpath',
        default=None,
        help='Train path')

    parser.add_argument(
        '-m', '--modelpath',
        default='model.json',
        help='model.yaml')

    parser.add_argument(
        '-a', '--augmentation',
        action='store_true',
        help='training on augmented data')

    parser.add_argument(
        '-c', '--context',
        action='store_true',
        help='context images included')

    parser.add_argument(
        '-bb', '--boundingbox',
        action='store_true',
        help='bounding box')

    parser.add_argument(
        '-ax', '--axis',
        default=0,
        type=int,
        help='Train path')




    args = parser.parse_args()
    if args.debug:
        ch.setLevel(logging.DEBUG)

    if args.boundingbox:
        bounding_box(args.trainpath, args.predictpath, args.modelpath, args.augmentation, args.context, args.configpath, args)

    if args.create_sample_data:
        sample_data(trainpath=args.trainpath)

    if args.train & args.predict:
        tr_history, slab =train(args.trainpath, args.modelpath, args.context, args.augmentation, config_file=args.configpath, axis=args.axis)
        val_history = validation(args.predictpath, args.modelpath, context=args.context, axis=args.axis)

        folders = glob.glob(os.path.join('experimenty/', '*/'))
        ints = [int(i.split('/')[-2]) for i in folders]
        hp = 'experimenty/'+ str(max(ints)+1) + '/'
        os.makedirs(hp)

        
        with open(hp + 'tr_history.yml', 'w') as f:
            yaml.dump(tr_history.history, f)
        with open(hp + 'val_history.yml', 'w') as f:
            yaml.dump(val_history, f)
        with open('config.txt') as f:
            lines = f.readlines()
            with open(hp + 'config.txt', "w") as f1:
                f1.writelines(lines)
    else:
        if args.train:
            train(args.trainpath, args.modelpath, args.augmentation, args.context, config_file=args.configpath, axis=args.axis)

        if args.predict:
            dt= predict(args.predictpath, args.modelpath, context=args.context, axis=args.axis)

    stats(args.configpath, 'stat.csv', slab, args, val_history[1])

if __name__ == "__main__":
    main()
