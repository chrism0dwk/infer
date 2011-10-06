#!/bin/bash

#$ -o /scratch/stsiab/$JOB_NAME.$JOB_ID.stdout
#$ -e /scratch/stsiab/$JOB_NAME.$JOB_ID.stderr
#$ -S /bin/bash
#$ -N fmdOccults
#$ -l h_vmem=500M,h_rt=12:00:00
#$ -cwd

. /etc/profile
module add gcc gsl ompi boost solstudio

echo Using $NSLOTS 1>&2

#collect -d /scratch/stsiab -M OPENMPI mpirun -np 16 -- ./fmdMcmc $HOME/fmdData/fmd2001_extract.csv $HOME/fmdData/fmd2001_ips.csv 10
time mpirun -output-filename /scratch/stsiab/$JOB_NAME.$JOB_ID ./fmdMcmc $HOME/fmdData/fmd2001_northeast_uk_centredM.csv /scratch/stsiab/FMD2001/sims/test/simTest1.trunc100.ipt $HOME/fmdData/testcovar.txt 110000