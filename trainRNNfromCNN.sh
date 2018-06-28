#!/bin/bash
''' ./runMelodyExtraction.sh "MODELDIM" paramFile.sh
'''
source $2
curname=`basename "$0"`
curdir="$( cd "$( dirname "${BASH_SOURCE[0] }" )" && pwd )"
expname=$1
expdir=$curdir/"Experiments"/$expname

if [ "$(ls $expdir)" ] ; then
  echo "dir exists"
else
  echo "creating dir"
  mkdir expdir
fi
if [ -z "$2" ] ; then
  paramfilename=$2
else
  paramfilename=$expdir/params
fi
cp $curdir/$1 $paramfilename
if [ $# -eq 3 ]; then
  python main.py --nOctave $nOctave --binsPerOctave $binsPerOctave --timeDepth $timeDepth --nHarmonics $nHarmonics --hopSize $hopSize --voicing $voicing --filterShape1 $filterShape1 --filterShape2 $filterShape2 --filterShape3 $filterShape3 --filterShape4 $filterShape4 --filterShape2D $filterShape2D --filterShape1D $filterShape1D --featureMaps1D $featureMaps1D --featureMaps2D $featureMaps2D --batchSize $batchSize --nEpochs $nEpochs --expName $expname --expDir $expdir --modelCnn $3 |& tee -a $expdir/out.log
fi
