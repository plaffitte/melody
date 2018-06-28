#!/bin/bash
''' Example run: ./runMelodyExtraction.sh "MODELDIM" paramFile.sh
'''
source $2
curname=`basename "$0"`
curdir="$( cd "$( dirname "${BASH_SOURCE[0] }" )" && pwd )"
expname=$1
expdir="/data/Experiments"/$expname
echo $expdir
paramfilename=$2
model=$3
# expdir=$curdir/"Experiments"/$expname

if [ "$(ls $expdir)" ] ; then
  echo "dir exists"
else
  echo "creating dir"
  mkdir $expdir
fi
python resumeTraining.py --nOctave $nOctave --binsPerOctave $binsPerOctave --timeDepth $timeDepth --nHarmonics $nHarmonics --hopSize $hopSize --voicing $voicing --batchSize $batchSize --nEpochs $nEpochs --expName $expname --seqNumber $seqNumber --stateFull $stateFull --expDir $expdir --expName $expname --model $3 |& tee $expdir/out.log
