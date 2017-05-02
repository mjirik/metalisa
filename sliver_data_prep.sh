#!/bin/bash
#PBS -N data_preparation
#PBS -l walltime=1:0:0
#PBS -l select=1:ncpus=2:mem=5gb:scratch_local=400mb
#PBS -q default
trap 'clean_scratch' TERM EXIT
cd /storage/plzen1/home/$USER/projects/metalisa
# module add python-2.7.6-gcc
# module add python27-modules-gcc
DIR=/storage/plzen1/home/$USER/miniconda2
export PYTHONPATH=$PYTHONPATH:$DIR/virtualenv/software/python-2.7.6/gcc/lib/python2.7/site-packages
source $DIR/bin/activate


PYTHONUSERBASE=$DIR
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python2.7/site-packages:$PYTHONPATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/storage/plzen1/home/mhlavac/keras_104/src
which python
python liver_data_preparation.py -f sliver -dd data/SLIVER
