#!/bin/bash
#PBS -N poloharezu
#PBS -l nodes=1:ppn=1:cl_konos
#PBS -q gpu
#PBS -l mem=20gb
#PBS -l walltime=5h
#PBS -l gpu=1
#trap 'clean_scratch' TERM EXIT

# data sdilene pres NFSv4
# /storage/plzen1/home/$USER

DATADIR="$HOME/data/medical/processed/metalisa/resolution100" 

module add python-2.7.6-gcc
module add cuda-7.5

# cd /storage/plzen1/home/tkolar/ Nahradit svým adresářem
cd /storage/plzen1/home/$USER/projects/metalisa

DIR=$HOME/keras_104
export PYTHONPATH=$PYTHONPATH:$DIR/virtualenv/software/python-2.7.6/gcc/lib/python2.7/site-packages
source $DIR/keras-1.0.4/bin/activate

PYTHONUSERBASE=$DIR/keras-1.0.4/
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python2.7/site-packages:$PYTHONPATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/storage/plzen1/home/mhlavac/keras_104/src

#cp $DATADIR/train.hdf5 $SCRATCHDIR  || exit 1
#cp $DATADIR/test.hdf5 $SCRATCHDIR  || exit 1
#cp $DATADIR/ker74.py $SCRATCHDIR   ||exit 1
#cd $SCRATCHDIR || exit 2

python slice_classification_cnn.py -i $DATADIR  -t ./experimenty/15 >./experimenty/15/log.txt -a -c 
 
#cp vystup.txt $DATADIR 
#cp vahy74.hdf5 $DATADIR 
#cp problem.txt $DATADIR ||  export CLEAN_SCRATCH=True

