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


def train(trainpath, modelpath="model.hdf5", csvpath=None):
    if csvpath is None:
        csvfiles = glob.glob(os.path.join(trainpath, "*.csv")
        csvpath = csvfiles[0]
        if len(csvfiles) > 1:
            logger.error("There is more .csv files in train directory. Please select one with csvpath parameter.")

    pass

def predict(predictpath, modelpath="model.hdf5", csvpath="prediction.csv"):
    pass


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
