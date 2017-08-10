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


def sliver_preparation(datadirpath, output_datadirpath="output_data", res=100, ax=0, organ='liver'):
    csvpath = output_datadirpath + '/sliver_label_'+str(res)+'_'+str(ax)+'_'+organ+'.csv'
    stat = output_datadirpath + '/sliver_stat'+str(res)+'_'+str(ax)+'_'+organ+'.csv'
    # datadirpath = '/home/trineon/projects/metalisa/data/SLIVER'
    f = h5py.File(output_datadirpath + '/sliver_' + str(res) + '_' + str(ax) + '_' + organ +'.hdf5', 'a')
    num = 1
    for image in glob.glob(datadirpath + '/*orig*.mhd'):
        group = f.create_group(image.split('/')[-1])
        orig, _ = DR.read(image)
        if ax != 0:
            orig = np.rollaxis(orig, ax)
        i = orig.shape[0]
        orig = misc.resize_to_shape(orig, [i, res, res])
        DW.write(orig, output_datadirpath + '/sliver_' +str(num)+ '_' + str(res)+'_' + str(ax)+ '.pklz', metadata={"voxelsize_mm": [1, 1, 1]})
        filename = output_datadirpath + '/sliver' +str(num) +'_' + str(res)+ '_' + str(ax)+ '.pklz'
        num += 1
        seg = image.replace('orig','seg')
        lab, _ = DR.read(seg)
        if ax != 0:
            lab = np.rollaxis(lab, ax)
        l = list()
        a = 1
        for slice in lab:
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
        if l[-1] == 2:
            x = len(l)
        else:
            x = l.index(3)
        if l[0] == 2:
            y = 0
        else:
            y = l.index(2)
        dt = {'filename': [filename, filename, filename], 'label': ['under ' + organ, organ, 'above ' + organ],
              'start_slice_number': [0, y, x],
              'stop_slice_number': [y - 1, x - 1, len(l)-1], 'axis': ax}
        if dt['stop_slice_number'][0] == -1:
            dt['stop_slice_number'][0] = 0
        if os.path.exists(csvpath):
            new_df = pd.read_csv(csvpath)
            df = pd.DataFrame.from_dict(dt)
            new_df = pd.concat([new_df, df], ignore_index=True)
        else:
            df0 = pd.DataFrame.from_dict(dt)
            new_df = df0
        new_df.to_csv(csvpath, index=False)
        a = y
        b = x-y
        c = len(l)-x
        dt = {'filename': [filename], 'under liver': [a] , 'liver': [b], 'above liver': [c], 'slices':[len(l)]}
        if os.path.exists(stat):
            new_df = pd.read_csv(stat)
            df = pd.DataFrame.from_dict(dt)
            new_df = pd.concat([new_df, df], ignore_index=True)
        else:
            df0 = pd.DataFrame.from_dict(dt)
            new_df = df0
        new_df.to_csv(stat, index=False)

    pass


def ircad_group(datadirpath, organ='liver'):
    # datadirpath = '/home/trineon/projects/metalisa/data/IRCAD'
    for folder in glob.glob(datadirpath + '/labels/*/'+organ+'/'):
        name = folder.split('/')[-3]
        if (folder + 'IRCAD_' + str(name) + '_' + organ +'.pklz') in glob.glob(folder+'*'):
            continue


        else:
            # concatenate CT slicis to one 3D ndarray [number_of slices, res(1), res(2)]
            scan = [None]* len(glob.glob(folder + '*'))
            for image in glob.glob(folder + '*'):
                label, _ = DR.read(image)
                scan[int(image.split('/')[-1].split('_')[-1])] = label
            scan = np.array(scan).astype(np.int32)
            scan = scan.squeeze()
            DW.write(scan, folder + 'IRCAD_' +  str(name) + '_' + organ + '.pklz',
                     metadata={"voxelsize_mm": [1, 1, 1]})


    pass
def ircad_preparation(datadirpath, output_datadirpath="output_data", organ="liver",res=100, ax=0):

    #test
    stat = output_datadirpath+'/stat_ircad'+str(res)+'_'+str(ax)+'_'+organ+'.csv'
    csvpath = output_datadirpath+'/label_ircad_'+str(res)+'_'+str(ax)+'_'+organ+'.csv'
    # datadirpath = '/home/trineon/projects/metalisa/data/IRCAD'

    seznam = [None] * 20
    for folder in glob.glob(datadirpath+'/Pacient/*/'):
        count = len(glob.glob(folder+'*'))
        l = [None] * count
        for image in glob.glob(folder+'*'):
            number = int(image.split('/')[-1].split('_')[-1])-1
            l[number], _ = DR.read(image)
            if ax != 0:
                l[number] = np.rollaxis(l[number], ax)
        for ind, i in enumerate(l):
            l[ind] = misc.resize_to_shape(i, [1, res, res])
        scan = np.array(l)
        if ax != 0:
            np.rollaxis(scan, ax)
        name = folder.split('/')[-2]
        scan = scan.squeeze()
        DW.write(scan, output_datadirpath + '/IRCAD_' +str(name) +'_' + str(res)+'_' + str(ax)+'.pklz', metadata={"voxelsize_mm": [1, 1, 1]})
        seznam[int(name)-1] = output_datadirpath + '/IRCAD_'+str(name) +'_' + str(res)+'_' + str(ax)+'.pklz'

    ll = [None] * 20
    for folder in glob.glob(datadirpath + '/labels/*/'+organ+'/'):
        count = len(glob.glob(folder+'*'))
        sez = list()
        for image in glob.glob(folder+'IRCAD*.pklz'):
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
        ll[int(folder.split('/')[-3])-1] = l
        file = seznam[int(folder.split('/')[-3])-1]

        if l[-1] == 2:
            x = len(l)
        else:
            x = l.index(3)
        if l[0] == 2:
            y = 0
        else:
            y = l.index(2)
        dt = {'filename': [file, file, file], 'label': ['under ' + organ, organ, 'above ' + organ],
              'start_slice_number': [0, y, x],
              'stop_slice_number': [y - 1, x - 1, len(l) - 1], 'axis': ax}
        if dt['stop_slice_number'][0] == -1:
            dt['stop_slice_number'][0] = 0
        if os.path.exists(csvpath):
            new_df = pd.read_csv(csvpath)
            df = pd.DataFrame.from_dict(dt)
            new_df = pd.concat([new_df, df], ignore_index=True)
        else:
            df0 = pd.DataFrame.from_dict(dt)
            new_df = df0
        new_df.to_csv(csvpath, index=False)
        a = y
        b = x - y
        c = len(l) - x
        dt = {'filename': [file], 'under liver': [a], 'liver': [b], 'above liver': [c], 'slices': [len(l)]}
        if os.path.exists(stat):
            new_df = pd.read_csv(stat)
            df = pd.DataFrame.from_dict(dt)
            new_df = pd.concat([new_df, df], ignore_index=True)
        else:
            df0 = pd.DataFrame.from_dict(dt)
            new_df = df0
        new_df.to_csv(stat, index=False)
    output_datadirpath + '/sliver_' + str(res) + '_' + str(ax) + '_' + organ + '.hdf5'
    f = h5py.File(output_datadirpath+'/sliver_'+str(res)+ '_' + str(ax) + '_' + str(organ) + '.hdf5', 'a')
    for index, file in enumerate(seznam):
        group = f.create_group(file.split('/')[-1])
        l = ll[index]
        scan, _ = DR.read(file)
        for ind, slice in enumerate(scan):
            name = str(ind)
            dset = group.create_dataset(name, data=slice)
            dset.attrs['teacher'] = l[ind]
            dset.attrs['origin file'] = file
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
    parser.add_argument('-od', '--output_dir', default='output_data')
    parser.add_argument('-r', '--resolution', default=100, type=int)
    parser.add_argument('-o', '--organ', default='liver')
    parser.add_argument('-a', '--axis', default=0, type=int)
    parser.add_argument('-bb', '--boundingbox', action='store_true')

    args = parser.parse_args()

    # test
    # args.function = 'ircad'


    if args.debug:
        ch.setLevel(logging.DEBUG)
    if args.boundingbox:
        if args.function == 'ircad':
            ircad_group(args.data_dir, organ=args.organ)
            for i in range(3):
                ircad_preparation(args.data_dir, args.output_dir ,res=args.resolution, organ=args.organ, ax=i)
        if args.function == 'sliver':
            for i in range(3):
                sliver_preparation(args.data_dir, args.output_dir, res=args.resolution, ax=i)
    else:
        if args.function == 'ircad':
            ircad_group(args.data_dir, organ=args.organ)
            ircad_preparation(args.data_dir, args.output_dir, res=args.resolution, organ=args.organ, ax=args.axis)
        if args.function == 'sliver':
            sliver_preparation(args.data_dir, args.output_dir, res=args.resolution, ax=args.axis)


if __name__ == "__main__":
    main()
