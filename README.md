# Princpal Component Analysis on ADC time serie

This code proposes to try to reduce correlation at large time, consequence of low frequency noise (known as flicker 
noise in electronics) using principal component analysis (PCA). 

## Presentation and motivation for a PCA
The final observable of interest is the averaged count over Ndim ADC count values and its precision (RMS). In case of no
correlation, averaging over Ndim samples improve the RMS by a factor 1/sqrt(Ndim) with respect to RMS of single counts. 
However, due to correlation this gain is not observed and the goal of the PCA is remove this correlation to recover the
expected precision gain in case of uncorrelated variables.

## Algorithm application

The code analyzes time serie of 9 millions ADC counts and group the values in arrays of Ndim. This sample
of 9000 observations is split into two, one being used for the auto-correlation matrix and the other to
test the decorrelation process. The PCA is then performed on 4500 Ndim vectors. Finally, the averaged is computed in 
the decorrelated space and its RMS is compared with the basic measurement for both training and testing samples.

## Results

The figure below shows the coefficients in the original space of the 6th first principal components (PC), 
the explained RMS fraction for each PC and the comparison of the final observable. The two first plots show
that we indeed have a low freqency noise since each PC seems to be harmonic (!) and the lowest frequency 
corresponds to the higher RMS.

![OCA result](https://github.com/rmadar/ADCTimeSeriePCA/blob/master/PCA_rResult.png)
