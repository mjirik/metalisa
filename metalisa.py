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
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout


def load_data(filename, csv=None):
    datap = io3d.datareader.read(filename, dataplus_format=True)
    data3d = datap["data3d"]

    if csv:
        df = pd.read_csv(csv)
        labs = np.unique(df["label"])
        vals = range(len(labs))

        slab = dict(zip(labs, vals))

        fn_unique = np.unique(df["filename"])

        name = filename.split('/')[-1]

        index = [i for i, item in enumerate(fn_unique) if item.endswith(name)][0]

        this_fn = fn_unique[index]
        one_file = df[df["filename"] == this_fn]

        maximum_slice_number = np.max(np.max(one_file[["start_slice_number", "stop_slice_number"]]))

        labels = [None] * maximum_slice_number

        dct = one_file[["label", "start_slice_number", "stop_slice_number"]].to_dict()

        for i in range(len(dct["start_slice_number"])):
            for i in slab.values():
                for j in range(dct["start_slice_number"][i], dct["stop_slice_number"][i]):
                    labels[j] = dct["label"][i]

    if len(data3d.shape) == 3:
        images = list()
        for i in data3d:
            images.append(i)

    else:
        images = data3d



    return images, labels #,slab


def train(trainpath, modelpath="model.hdf5", csvpath=None):
    if csvpath is None:
        csvfiles = glob.glob(os.path.join(trainpath, "*.csv"))
        csvpath = csvfiles[0]
        if len(csvfiles) > 1:
            logger.error("There is more .csv files in train directory. Please select one with csvpath parameter.")

    tl = list()
    td = list()

    for filename in glob.glob(os.path.join(trainpath, '*.pklz')):
        logger.info(filename)
        image, labels = load_data(filename, csvpath)
        td += image
        tl += labels

    td = np.array(tl)
    shape = td.shape
    td.reshape(shape[0], 1, shape[1], shape[2])
    tl = np.array(td)


    model = Sequential()
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

    model.train(td, tl, batch_size=32, nb_epoch=100)

    with open(modelpath, 'w') as json_file:
        model_json = model.to_json()
        json_file.write(model_json)

def predict(predictpath, modelpath="model.json", csvpath="prediction.csv"):

    json_file = open(modelpath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    vysledky = list()
    names = list()

    for filename in glob.glob(os.path.join(predictpath, '*.tiff')):
        logger.info(filename)
        image = load_data(filename)
        for i in image:
            vysledky.append(model.predict(image))
            names.append(filename)
    with open(csvpath, 'w') as f:
        if os.path.exists(csvpath):
            df0 = pd.read_csv(csvpath)
        else:
            df0 = pd.DataFrame()
        new_df = df0
        for i in range(len(vysledky)):
            df = pd.DataFrame([names[i]], [vysledky[i]], columns=['filename','result'])
            new_df = new_df.append(df)
        new_df.to_csv(csvpath)

def sample_data(trainpath):
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
        "label": ["under liver", "liver", "above liver"],
        "start_slice_number": [0, start_slice_number, stop_slice_number],
        "stop_slice_number": [start_slice_number - 1, stop_slice_number - 1, datasize[0]],
        "filename": [None, None, None]
    }


    return data3d, metadata


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
        '-c', '--configpath',
        default=None,
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
        default=None,
        help='model.yaml')

    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)


    if args.configpath is None:
        pass

    if args.create_sample_data:
        sample_data(trainpath=args.trainpath)

    pass

    if args.train:
        train(args.trainpath, args.modelpath)

    if args.predict:
        predict(args.predictpath, args.modelpath)


if __name__ == "__main__":
    main()
