#PBS -N poloharezu_kontext
#PBS -l walltime=10:0:0
#PBS -l select=1:ncpus=2:mem=2gb:scratch_local=400mb:cl_konos=True:ngpus=2
#PBS -q gpu
trap 'clean_scratch' TERM EXIT

# data sdilene pres NFSv4
# /storage/plzen1/home/$USER

DATADIR="$HOME/projects/metalisa/output_data/ircad/" 
DATADIR2="$HOME/projects/metalisa/output_data/sliver2/" 
Mpath="$HOME/projects/metalisa/model_c.json"
module add python-2.7.6-gcc
module add cuda-7.5

# cd /storage/plzen1/home/tkolar/ Nahradit svým adresářem
cd /storage/plzen1/home/$USER/projects/metalisa/

DIR=$HOME/miniconda2
# export PYTHONPATH=$PYTHONPATH:$DIR/virtualenv/software/python-2.7.6/gcc/lib/python2.7/site-packages
source $DIR/bin/activate

#PYTHONUSERBASE=$DIR/keras-1.0.4/
#export PATH=$PYTHONUSERBASE/bin:$PATH
#export PYTHONPATH=$PYTHONUSERBASE/lib/python2.7/site-packages:$PYTHONPATH

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/storage/plzen1/home/mhlavac/keras_104/src

#cp $DATADIR/train.hdf5 $SCRATCHDIR  || exit 1
#cp $DATADIR/test.hdf5 $SCRATCHDIR  || exit 1
#cp $DATADIR/ker74.py $SCRATCHDIR   ||exit 1
#cd $SCRATCHDIR || exit 2

python metalisa.py -tp $DATADIR  -t -pp $DATADIR2 -p -m $Mpath -c -a
 
#cp vystup.txt $DATADIR 
#cp vahy74.hdf5 $DATADIR 
#cp problem.txt $DATADIR ||  export CLEAN_SCRATCH=True

