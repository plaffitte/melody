### -------------------->>> PARAM DEFINTION <<<-------------------- ###
### Data Params:
# fftSize="360"
binsPerOctave="12"
nOctave="6"
timeDepth="25"
nHarmonics="6"
hopSize="1"
voicing="True"
stateFull="True"
seqNumber="10"

### Model params:
filterShape1="[1,1,6]"    # Filters' shape (f, t, h)
filterShape2="[1,1,4]"
filterShape3="[1,1,2]"
filterShape4="[1,1,1]"
filterShape2D="[5,5,2]"
filterShape1D="[1,1,6]"
featureMaps1D="128"
featureMaps2D="128"
batchSize="50"
nEpochs="500"