# Principal Component Analysis on ADC time serie

This code proposes to try to reduce correlation at large time, consequence of low frequency noise (known as flicker 
noise in electronics) using principal component analysis (PCA). 

## Presentation and motivation for a PCA
The final observable of interest is the averaged count over Ndim ADC count values and its precision (RMS). In case of no
correlation, averaging over Ndim samples improve the RMS by a factor 1/sqrt(Ndim) with respect to RMS of single counts. 
However, due to correlation this gain is not observed and the goal of the PCA is remove this correlation to recover the
expected precision gain in case of uncorrelated variables.

## Algorithm application

The code analyzes time serie of 9 millions ADC counts and group the values in arrays of Ndim=1000. This sample
of 9000 observations is split into two, one being used for the auto-correlation matrix and the other to
test the decorrelation process. The PCA is then performed on 4500 Ndim vectors. Finally, the averaged is computed in 
the decorrelated space and its RMS is compared with the basic measurement for both training and testing samples.

## Results
The figure below shows the correlation before PCA and after, as well as a scatter plot between first and fiftyth variables
(ADC values before and PC after transformation).

![PCA decorrelation](https://github.com/rmadar/ADCTimeSeriePCA/blob/master/Correlations.png)

The figure below shows the ADC time serie, the coefficients in the original space of the 6th first principal components (PC), 
the explained RMS fraction for each PC and the comparison of the final observable. The two plots top-right and bottom-left 
show that we indeed have a low freqency noise since each PC seems to be harmonic (!) and the lowest frequency 
corresponds to the higher RMS.

![PCA result](https://github.com/rmadar/ADCTimeSeriePCA/blob/master/Results.png)

## Go results

To run, once the [Go](https://golang.org/doc/install) toolchain has been installed:

```sh
$> go get github.com/rmadar/ADCTimeSeriePCA
$> time $GOPATH/bin/ADCTimeSeriePCA
pca: loading data...
pca: ==> done (1.639700776s)
pca: 8996 events of 1000 dimension multiplet
pca: splitting data sample into training (50%) and testing (50%)
pca: ==> done (1.239µs)
pca: diagonalizing PCA matrix with training data...
pca: ==> done (25.703314369s)
pca: decorrelate variable and averaged values...
pca: ==> done (3.382069495s)
pca: creating plots...
pca: ==> done (2.597800932s)
pca: creating correlation plots...
pca: ==> done (8.505984618s)

real	0m41.856s
user	1m11.430s
sys	0m0.270s
```

![Go PCA decorrelation](https://github.com/rmadar/ADCTimeSeriePCA/blob/master/go-correlations.png)
![Go PCA result](https://github.com/rmadar/ADCTimeSeriePCA/blob/master/go-results.png)



compared to, with Python2:
```sh
$> time python2 ./PCAdecorrelation.py

--> Data being loaded ...
   * done in 1.28s: 8996 events of 1000-dimension multiplet

--> Data sample is splitted into training (50%) and testing (50%) samples ...
   * done in 0.00s

--> PCA matrix is being diagonalized using training data ...
   * done in 22.93s

--> Decorrelate variable and averaged values ...
   * done in 12.39s

--> Plots are being made ...
   * Results done in 5.85s
   * Correlations done in 13.76s


real	1m6.176s
user	0m54.201s
sys	0m0.580s
```
