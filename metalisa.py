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


def train(trainpath, modelpath="model.hdf5", csvpath=None):
    if csvpath is None:
        csvfiles = glob.glob(os.path.join(trainpath, "*.csv"))
        csvpath = csvfiles[0]
        if len(csvfiles) > 1:
            logger.error("There is more .csv files in train directory. Please select one with csvpath parameter.")

    pass

def predict(predictpath, modelpath="model.hdf5", csvpath="prediction.csv"):
    pass

def sample_data(trainpath):
    filename = os.path.join(trainpath, "data01.pklz")
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
        '-l', '--logfile',
        default="~/teigen.log",
        help='Debug mode')
    parser.add_argument(
        '-t', '--trainpath',
        default=None,
        help='Train path')

    parser.add_argument(
        '-p', '--predictpath',
        default=None,
        help='Train path')

    parser.add_argument(
        '-m', '--modelpath',
        default=None,
        help='model.yaml')

    args = parser.parse_args()


    if args.debug:
        ch.setLevel(logging.DEBUG)

    if args.parameterfile is None:
        pass

    if args.trainpath is not None:
        train(args.trainpath, args.modelpath)

    if args.predictpath is not None:
        predict(args.predictpath, args.modelpath)

if __name__ == "__main__":
    main()
