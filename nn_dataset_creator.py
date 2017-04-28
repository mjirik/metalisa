import os
import io3d
import h5py
import sed3
import numpy as np
import glob
import argparse
global f
import pandas as pd
import imtools.misc as misc


def differencing(path, res, axis):
    sez = list()
    teacher = list()
    dr = io3d.DataReader()
    datap = dr.Get3DData(path, dataplus_format=True)
    ed = sed3.sed3(datap['data3d'])
    datap['data3d'] = misc.resize_to_shape(datap['data3d'], [datap['data3d'].shape[0],100,100])
    ed.show()

    try:
        under = np.where(ed.seeds == 1)[0][0]
    except:
        under = None
    try:
        over = np.where(ed.seeds == 3)[0][0]
    except:
        over = None
    number_of_slice = np.shape(ed.seeds)[0]
    c = list()
    if under is None and over is None:
        return [],[]
    elif under is None or over is None:
        print u'byla zadana jenom jedna mez zadejte obe nebo zadnou'
        w, y = differencing(path, res)
        return w, y
    if over > under:
        c = [1 for i in range(under)]
        b = [2 for i in range(over-under)]
        c.extend(b)
        b = [3 for i in range(number_of_slice-over)]
        c.extend(b)
    else:
        c = [3 for i in range(over)]
        b = [2 for i in range(under - over)]
        c.extend(b)
        b = [1 for i in range(number_of_slice - under)]
        c.extend(b)
        over, under = under, over
    for i in range(number_of_slice):
        a = datap['data3d'][i,:,:]
        x = c[i]
        teacher.append(x)
        sez.append(a)

    return sez, teacher, under, over


def main():
    global f
    parser = argparse.ArgumentParser(description=
                                     'Creating dataset for Convolutional neural network')
    parser.add_argument('-i', '--datadir',
                    default=None,
                    help='path to data dir')
    parser.add_argument('-o', '--output',
                        default=None,
                        help='path to output file')
    parser.add_argument('-r', '--resolution', default='40', help='resolution of slices')

    parser.add_argument('-cc','--create_csv_file', action='store_true')
    parser.add_argument('-a','--axis',default=0)
    parser.add_argument('-c','--csv_file', default='labels.csv')
    parser.add_argument('-org','--organ',default='liver')

    args = parser.parse_args()

    if args.datadir:
        path = args.datadir
    else:
        # path = '/home/trineon/neuronovka/train/test-part2'
        path = '/home/trineon/neuronovka/test/test-part1' #test
        # path = '/home/trineon/neuronovka/train/training'
    if args.output:
        out = args.output
    else:
        out = '/home/trineon/neuronovka/test100.hdf5'
        # out = '/home/trineon/neuronovka/train100.hdf5'


    f = h5py.File(out, 'a')
    for filename in glob.glob(os.path.join(path, '*.mhd')):
        res = 100
        s, t, under, over = differencing(filename, res, args.axis)
        if s == []:
            print 'preskoceno'
            continue
        try:
            group = f.create_group(filename.split('/')[-1])
        # except ValueError:
        #     print filename + ' jiz v datasetu je'
        #     continue
        except:
            group = f.create_group(filename.split('/')[-1]+'2')
        for ind, slice in enumerate(s):
            name = str(ind)
            dset = group.create_dataset(name,data = slice)
            dset.attrs['teacher'] = t[ind]
            dset.attrs['origin file'] = filename
        if args.csv:
            dt = {'filename': [filename, filename, filename], 'label': ['under ' + args.organ, args.organ,
                                                                        'above ' + args.organ],
                  'start_slice_number': [0, under, over],
                  'stop_slice_number': [under - 1, over - 1, len(t) - 1], 'axis': args.axis}
            if os.path.exists(args.csv):
                new_df = pd.read_csv(args.csv)
                df = pd.DataFrame.from_dict(dt)
                new_df = pd.concat([new_df, df], ignore_index=True)
            else:
                df0 = pd.DataFrame.from_dict(dt)
                new_df = df0
            new_df.to_csv(args.c, index=False)
    f.close()


if __name__ == "__main__":
    main()