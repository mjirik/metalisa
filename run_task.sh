#!/bin/bash
#qsub -l walltime=2h -l nodes=1:ppn=2:gpu=1,mem=4gb,scratch=100mb uloha1.sh
qsub -l walltime=2h -l nodes=1:ppn=2:gpu=0,mem=4gb,scratch=100mb uloha1.sh

