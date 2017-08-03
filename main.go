package main

import (
	"bufio"
	"log"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

const (
	Ndim   = 1000
	fTrain = 0.5
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("pca: ")

	t1 := time.Now()
	log.Printf("loading data...")
	dataRaw, err := readRawData("data.txt")
	if err != nil {
		log.Fatal(err)
	}
	rows, cols := dataRaw.Dims()
	mean := mat.Sum(dataRaw) / float64(rows*cols)
	log.Printf("mean=%v", mean)

	data0 := mat.NewDense(rows, cols, nil)
	data0.Clone(dataRaw)
	data0.Apply(func(i, j int, v float64) float64 {
		return v - mean
	}, data0)

	log.Printf("==> done (%v)", time.Since(t1))
	log.Printf("%d events of %d dimension multiplet", rows, cols)

	log.Printf("splitting data sample into training (%2.0f%%) and testing (%2.0f%%)", fTrain*100, (1-fTrain)*100)
	t2 := time.Now()
	nTrain := int(float64(rows+1) * fTrain)
	nTest := rows - int(float64(rows)*fTrain+1)
	var (
		data0Train = data0.Slice(0, nTrain, 0, Ndim)
		data0Test  = data0.Slice(rows-nTest, rows, 0, Ndim)
	)
	log.Printf("==> done (%v)", time.Since(t2))

	log.Printf("diagonalizing PCA matrix with training data...")
	var pca stat.PC

	t3 := time.Now()
	ok := pca.PrincipalComponents(data0Train, nil)
	if !ok {
		log.Fatalf("error computing principal components")
	}
	log.Printf("==> done (%v)", time.Since(t3))

	log.Printf("decorrelate variable and averaged values...")

	t4 := time.Now()
	var (
		dataDecorTrain mat.Dense
		dataDecor      mat.Dense
	)

	svd := pca.VectorsTo(nil)
	dataDecorTrain.Mul(data0Train, svd)
	dataDecor.Mul(data0Test, svd)

	meanInit := mat.NewDense(rows, 1, nil)
	for irow := 0; irow < rows; irow++ {
		row := dataRaw.RowView(irow)
		meanInit.Set(irow, 0, mat.Sum(row)/float64(Ndim))
	}

	meanDecor := mat.NewDense(nTest, 1, nil)
	for irow := 0; irow < nTest; irow++ {
		row := dataDecor.RowView(irow)
		meanDecor.Set(irow, 0, mat.Sum(row)/float64(Ndim)+mean)
	}

	meanDecorTrain := mat.NewDense(nTrain, 1, nil)
	for irow := 0; irow < nTrain; irow++ {
		row := dataDecorTrain.RowView(irow)
		meanDecorTrain.Set(irow, 0, mat.Sum(row)/float64(Ndim)+mean)
	}
	log.Printf("==> done (%v)", time.Since(t4))
}

func readRawData(fname string) (*mat.Dense, error) {
	f, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var adc []float64
	s := bufio.NewScanner(f)
	for s.Scan() {
		txt := s.Text()
		v, err := strconv.Atoi(txt)
		if err != nil {
			return nil, err
		}
		adc = append(adc, float64(v))
	}

	ndim := len(adc) / Ndim
	adc = adc[:ndim*Ndim]
	m := mat.NewDense(ndim, Ndim, adc)
	return m, nil
}
