#!/bin/bash

#$ -S /bin/bash
#$ -N testMcmc
#$ -cwd

. /etc/profile
module add gcc gsl acml ompi

time mpirun ./testMcmc ../../../testdata/aiPopulation.csv ../../../testdata/sim702.ipt
