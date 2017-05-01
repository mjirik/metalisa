#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import logging, os
import h5py
import glob
import numpy as np
import io3d.datareader as DR
import io3d.datawriter as DW
import argparse
import pandas as pd
import imtools.misc as misc


logger = logging.getLogger(__name__)


def sliver_preparation(datadirpath, output_datadirpath="output_data", res=100, ax=0):
    organ = 'liver'
    csvpath = output_datadirpath + '/sliver_label_'+str(res)+'.csv'
    # datadirpath = '/home/trineon/projects/metalisa/data/SLIVER'
    f = h5py.File(output_datadirpath +'/sliver_'+str(res)+'.hdf5', 'a')
    name = 1
    for image in glob.glob(datadirpath + '/*orig*.mhd'):
        group = f.create_group(image.split('/')[-1])
        orig, _ = DR.read(image)
        orig = misc.resize_to_shape(orig, [1, res, res])
        if ax != 0:
            orig = np.rollaxis(orig, ax)
        DW.write(orig, output_datadirpath + '/sliver' +str(name) +'_'+str(ax)+'_' + str(res)+'.vtk', metadata={"voxelsize_mm": [1, 1, 1]})
        filename = output_datadirpath + '/sliver' +str(name) +str(ax)+'_'  +'_' + str(res)+'.vtk'
        name += 1
        seg = image.replace('orig','seg')
        lab, _ = DR.read(seg)
        if ax != 0:
            lab = np.rollaxis(lab, ax)
        l = list()
        a = 1
        for slice in lab:
            print np.unique(slice)
            if len(np.unique(slice)) > 1:
                l.append(2)
                a = 2
            else:
                if a == 2:
                    l.append(3)
                else:
                    l.append(1)
        del lab
        for ind, slice in enumerate(orig):
            name = str(ind)
            dset = group.create_dataset(name, data=slice)
            dset.attrs['teacher'] = l[ind]
            dset.attrs['origin file'] = filename
        dt = {'filename': [filename, filename, filename], 'label': ['under ' + organ, organ, 'above ' + organ],
              'start_slice_number': [0, l.index(2), l.index(3)],
              'stop_slice_number': [l.index(2) - 1, l.index(3) - 1, len(l)-1], 'axis': ax}
        if os.path.exists(csvpath):
            new_df = pd.read_csv(csvpath)
            df = pd.DataFrame.from_dict(dt)
            new_df = pd.concat([new_df, df], ignore_index=True)
        else:
            df0 = pd.DataFrame.from_dict(dt)
            new_df = df0
        new_df.to_csv(csvpath, index=False)


    pass

def ircad_group(datadirpath, organ='liver'):
    datadirpath = '/home/trineon/projects/metalisa/data/IRCAD'

    for folder in glob.glob(datadirpath + '/labels/*/'+organ+'/'):
        name = folder.split('/')[-3]
        if (folder + 'IRCAD_' + str(name) +'_' +  organ +'.vtk') in glob.glob(folder+'*'):
            return

        else:
            scan = [None]* len(glob.glob(folder + '*'))
            for image in glob.glob(folder + '*'):
                label, _ = DR.read(image)
                scan[int(image.split('/')[-1].split('_')[-1])] = label
            scan = np.array(scan).astype(np.int32)
            print np.unique(scan)
            DW.write(scan, folder + 'IRCAD_'  + str(name) +'_' +  organ +'.vtk',
                     metadata={"voxelsize_mm": [1, 1, 1]})
            lab,_ = DR.read(folder + 'IRCAD_'  + str(name) +'_' +  organ +'.vtk')
            print np.unique(lab)

def ircad_preparation(datadirpath, output_datadirpath="output_data", organ="liver",res=100, ax=0):

    #test
    csvpath = output_datadirpath+'/label_ircad.csv'
    datadirpath = '/home/trineon/projects/metalisa/data/IRCAD'
    seznam = [None] * 20
    for folder in glob.glob(datadirpath+'/Pacient/*/')[:2]:
        count = len(glob.glob(folder+'*'))
        l = [None] * count
        for image in glob.glob(folder+'*'):
            number = int(image.split('/')[-1].split('_')[-1])-1
            l[number], _ = DR.read(image)
        for ind, i in enumerate(l):
            l[ind] = misc.resize_to_shape(i, [1, res, res])
        scan = np.array(l)
        if ax != 0:
            np.rollaxis(scan, ax)
        name = folder.split('/')[-2]
        DW.write(scan, output_datadirpath + '/IRCAD_'+organ+'_'+str(ax)+'_' +str(name) +'_' + str(res)+'.vtk', metadata={"voxelsize_mm": [1, 1, 1]})
        seznam[int(name)-1] = output_datadirpath + '/IRCAD_'+organ+'_'+str(ax)+'_' +str(name) +'_' + str(res)+'.vtk'

    for folder in glob.glob(datadirpath + '/labels/*/'+organ+'/')[:2]:
        count = len(glob.glob(folder+'*'))
        sez = list()
        for image in glob.glob(folder+'IRCAD*.vtk'):
            label, _ = DR.read(image)
        if ax != 0:
            label = np.rollaxis(label, ax)
        l = list()
        a = 1
        for slice in label:
            if len(np.unique(slice)) > 1:
                l.append(2)
                a = 2
            else:
                if a == 2:
                    l.append(3)
                else:
                    l.append(1)
        file = seznam[int(folder.split('/')[-3])-1]
        dt = {'filename': [file,file,file], 'label':['under '+ organ, organ, 'above '+ organ],
              'start_slice_number': [0, l.index(2), l.index(3)],
              'stop_slice_number': [l.index(2) - 1, l.index(3) - 1, len(l)-1], 'axis': ax}
        if os.path.exists(csvpath):
            new_df = pd.read_csv(csvpath)
            df = pd.DataFrame.from_dict(dt)
            new_df = pd.concat([new_df, df], ignore_index = True)
        else:
            df0 = pd.DataFrame.from_dict(dt)
            new_df = df0
        new_df.to_csv(csvpath, index=False)

    f = h5py.File(output_datadirpath+'/IRCAD_'+str(res)+'.hdf5', 'a')
    for i in seznam:
        group = f.create_group(i.split('/')[-1])
        scan, _ = DR.read(i)
        for ind, slice in enumerate(scan):
            name = str(ind)
            dset = group.create_dataset(name, data=slice)
            dset.attrs['teacher'] = l[ind]
            dset.attrs['origin file'] = i
    f.close()
    pass

def interactive_preparation(datadirpath, output_datadirpath="output_data"):

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
    parser.add_argument(
        '-p', '--parameterfile',
        default=None,
        # required=True,
        help='input parameter file'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')

    parser.add_argument(
        '-ni', '--nointeractivity', action='store_true',
        help='No interactivity mode')

    parser.add_argument(
        '-l', '--logfile',
        default="~/teigen.log",
        help='Debug mode')
    parser.add_argument('-f', '--function', default='manual', help='manual/ircad/sliver')
    parser.add_argument('-dd', '--data_dir')
    parser.add_argument('-od', '--output_dir')
    parser.add_argument('-r', '--resolution', default=100, type=int)
    parser.add_argument('-o', '--organ', default='liver')
    parser.add_argument('-a', '--axis', default=0, type=int)
    args = parser.parse_args()

    # test
    args.function = 'ircad'


    if args.debug:
        ch.setLevel(logging.DEBUG)
    if args.function == 'ircad':
        ircad_group(args.data_dir, organ=args.organ)
        # ircad_preparation(args.data_dir,res=args.resolution, organ=args.organ, ax=args.axis)
    if args.function == 'sliver':
        sliver_preparation(args.data_dir,res = args.resolution, ax=args.axis)


if __name__ == "__main__":
    main()
