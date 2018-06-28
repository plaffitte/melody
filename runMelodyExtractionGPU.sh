#!/bin/bash
''' Example run: ./runMelodyExtraction.sh "MODELDIM" paramFile.sh
'''
source $2
curname=`basename "$0"`
curdir="$( cd "$( dirname "${BASH_SOURCE[0] }" )" && pwd )"
expname=$1
expdir="/data/anasynth_nonbp/laffitte/Experiments"/$expname
echo $expdir
# expdir=$curdir/"Experiments"/$expname

if [ "$(ls $expdir)" ] ; then
  echo "dir exists"
else
  echo "creating dir"
  mkdir $expdir
fi
paramfilename=$2
cp $curdir/$paramfilename $expdir/$paramfilename
if [ $# -eq 3 ]; then
  python -u main_gpu.py --nOctave $nOctave --binsPerOctave $binsPerOctave --timeDepth $timeDepth --nHarmonics $nHarmonics --hopSize $hopSize --voicing $voicing --batchSize $batchSize --nEpochs $nEpochs --expName $expname --seqNumber $seqNumber --stateFull $stateFull --expDir $expdir --model $3 |& tee $expdir/out.log
elif [ $# -eq 4 ]; then
  python -u main_gpu.py --nOctave $nOctave --binsPerOctave $binsPerOctave --timeDepth $timeDepth --nHarmonics $nHarmonics --hopSize $hopSize --voicing $voicing --batchSize $batchSize --nEpochs $nEpochs --expName $expname --seqNumber $seqNumber --stateFull $stateFull--expDir $expdir --modelCnn $3 --modelRnn $4 |& tee $expdir"/out.log"
else
  python -u main_gpu.py --nOctave $nOctave --binsPerOctave $binsPerOctave --timeDepth $timeDepth --nHarmonics $nHarmonics --hopSize $hopSize --voicing $voicing --batchSize $batchSize --nEpochs $nEpochs --expName $expname --seqNumber $seqNumber --stateFull $stateFull --expDir $expdir |& tee $expdir"/out.log"
fi
