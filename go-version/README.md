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

![Go PCA decorrelation](https://github.com/rmadar/ADCTimeSeriePCA/blob/master/go-version/go-correlations.png)
![Go PCA result](https://github.com/rmadar/ADCTimeSeriePCA/blob/master/go-version/go-results.png)



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
