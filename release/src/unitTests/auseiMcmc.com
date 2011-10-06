#!/bin/bash

#$ -S /bin/bash
#$ -N eiWithSIR
#$ -l h_vmem=500M,h_rt=6:0:0
#$ -cwd

. /etc/profile
module add gcc gsl ompi boost solstudio

echo Using $NSLOTS 1>&2

#collect -d /scratch/stsiab -M OPENMPI mpirun -np 16 -- ./fmdMcmc $HOME/fmdData/fmd2001_extract.csv $HOME/fmdData/fmd2001_ips.csv 10
time mpirun ./auseiMcmc $HOME/auseiData/ausei_covars_area_sf.csv $HOME/auseiData/ausei_ips_sf.csv 300000