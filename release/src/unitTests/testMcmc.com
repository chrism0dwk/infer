#!/bin/bash

#$ -S /bin/bash
#$ -N fmdMcmc
#$ -l h_vmem=300M,h_rt=6:00:00
#$ -cwd

. /etc/profile
module add gcc gsl ompi boost solstudio

echo Using $NSLOTS 1>&2

#collect -d /scratch/stsiab -M OPENMPI mpirun -np 16 -- ./fmdMcmc $HOME/fmdData/fmd2001_extract.csv $HOME/fmdData/fmd2001_ips.csv 10
time mpirun  ./fmdMcmc $HOME/fmdData/fmd2001_northeast_uk_centredM.csv /scratch/stsiab/FMD2001/sims/12/posteriorSim.com.91272.126002.ipt $HOME/fmdData/testcovar.txt 500000