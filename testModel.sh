#!/bin/bash
source $2
curname=`basename "$0"`
curdir="$( cd "$( dirname "${BASH_SOURCE[0] }" )" && pwd )"
expname=$1
expdir=$3
echo $expdir
# expdir=$curdir/"Experiments"/$expname

if [ "$(ls $expdir)" ] ; then
  echo "dir exists"
else
  echo "creating dir"
  mkdir $expdir
fi
paramfilename=$2
if [ $# -eq 4 ]; then
  python3.5 main.py --nOctave $nOctave --binsPerOctave $binsPerOctave --timeDepth $timeDepth --nHarmonics $nHarmonics --hopSize $hopSize --voicing $voicing --filterShape1 $filterShape1 --filterShape2 $filterShape2 --filterShape3 $filterShape3 --filterShape4 $filterShape4 --filterShape2D $filterShape2D --filterShape1D $filterShape1D --featureMaps1D $featureMaps1D --featureMaps2D $featureMaps2D --batchSize $batchSize --nEpochs $nEpochs --expName $expname --expDir $expdir --seqNumber $seqNumber --stateFull $stateFull --model $4 |& tee $expdir/out.log
elif [ $# -eq 5 ]; then
  python3.5 main.py --nOctave $nOctave --binsPerOctave $binsPerOctave --timeDepth $timeDepth --nHarmonics $nHarmonics --hopSize $hopSize --voicing $voicing --filterShape1 $filterShape1 --filterShape2 $filterShape2 --filterShape3 $filterShape3 --filterShape4 $filterShape4 --filterShape2D $filterShape2D --filterShape1D $filterShape1D --featureMaps1D $featureMaps1D --featureMaps2D $featureMaps2D --batchSize $batchSize --nEpochs $nEpochs --expName $expname --expDir $expdir --seqNumber $seqNumber --stateFull $stateFull --modelCnn $4 --modelRnn $5 |& tee $expdir/out.log
fi
