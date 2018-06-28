#!/bin/bash
''' Example run: ./runMelodyExtraction.sh "MODELDIM" paramFile.sh
'''
source $2
curname=`basename "$0"`
curdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
expdir=$1
echo $expdir

if [ "$(ls $expdir)" ] ; then
  echo "dir exists"
else
  echo "creating dir"
  mkdir $expdir
fi
paramfilename=$2
cp $curdir/$paramfilename $expdir/$paramfilename

python main.py --nOctave $nOctave --binsPerOctave $binsPerOctave --timeDepth $timeDepth --nHarmonics $nHarmonics --hopSize $hopSize --voicing $voicing --filterShape1 $filterShape1 --filterShape2 $filterShape2 --filterShape3 $filterShape3 --filterShape4 $filterShape4 --filterShape2D $filterShape2D --filterShape1D $filterShape1D --featureMaps1D $featureMaps1D --featureMaps2D $featureMaps2D --batchSize $batchSize --nEpochs $nEpochs --expName "TESTDEEPSALIENCE" --expDir $expdir |& tee $expdir/out.log
