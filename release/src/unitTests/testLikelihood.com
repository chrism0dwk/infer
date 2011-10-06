#!/bin/bash

#$ -S /bin/bash
#$ -N likelihood
#$ -l h_vmem=300M,h_rt=12:00:00
#$ -cwd

. /etc/profile
module add gcc gsl ompi boost valgrind

echo Using $NSLOTS 1>&2

#collect -d /scratch/stsiab -M OPENMPI mpirun -np 16 -- ./fmdMcmc $HOME/fmdData/fmd2001_extract.csv $HOME/fmdData/fmd2001_ips.csv 10
valgrind --tool=cachegrind --L2=4194304,16,64 --cachegrind-out-file=/scratch/stsiab/cachegrind-fmdMcmc --branch-sim=yes ./fmdMcmc $HOME/fmdData/fmd2001_northeast_uk_centredM.csv  $HOME/fmdData/fmd2001_northeast_uk_ips.csv   $HOME/fmdData/testcovar.txt 10