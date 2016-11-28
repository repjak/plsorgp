#!/bin/bash
#PBS -N plsorgp_benchmark
#PBS -l nodes=1#excl
#PBS -l mem=1gb
#PBS -l scratch=100mb
#PBS -l matlab=1
#PBS -l matlab_Optimization_Toolbox=1
#PBS -l matlab_Statistics_Toolbox=1
#PBS -l walltime=1h
#PBS -j oe
#PBS -m e

DATADIR="$PBS_O_WORKDIR"
LOGDIR=$DATADIR/logs

if [ ! -d $LOGDIR ]; then
  mkdir $LOGDIR
fi

cd $DATADIR

module add matlab

matlab_cmd="publish('synthTest.m', 'format', 'html'); publish('benchmark.m', 'format', 'html', 'showCode', false); exit(0);"

echo "exit(1);" | matlab -nosplash -nodisplay -nodesktop -r "$matlab_cmd" > $LOGDIR/$PBS_JOBID.out 2>&1
