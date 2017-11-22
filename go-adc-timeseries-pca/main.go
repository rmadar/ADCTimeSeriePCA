package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"go-hep.org/x/hep/hbook"
	"go-hep.org/x/hep/hplot"

	"golang.org/x/sync/errgroup"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	Ndim   = 1000
	fTrain = 0.5
)

var (
	colors = plotutil.SoftColors
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

	log.Printf("creating plots...")
	t5 := time.Now()
	tp := hplot.NewTiledPlot(draw.Tiles{Cols: 2, Rows: 2})

	var grp errgroup.Group
	grp.Go(func() error {
		const nToPlot = 100000
		p := tp.Plot(0, 0)
		p.Title.Text = fmt.Sprintf("ADC counts vs Time (%d events only)", nToPlot)
		p.X.Label.Text = "Time [µs]"
		p.Y.Label.Text = "ADC counts"
		adc := dataRaw.RawMatrix().Data
		xys := make(plotter.XYs, nToPlot)
		for i, adc := range adc[:nToPlot] {
			xys[i].X = float64(i)
			xys[i].Y = adc
		}
		pts, err := hplot.NewScatter(xys)
		if err != nil {
			return err
		}
		pts.Color = colors[2]
		pts.Radius = 0.2 * vg.Millimeter
		p.Add(pts)
		return nil
	})

	grp.Go(func() error {
		p := tp.Plot(0, 1)
		p.Title.Text = `PC $= \Sigma_{i} \: \alpha_{i} n_{i}$`
		p.X.Label.Text = "i^th sample (= Time [µs])"
		p.Y.Label.Text = `\alpha_i`
		p.Legend.Top = true
		for i := 0; i < 6; i++ {
			col := svd.ColView(i)
			xys := make(plotter.XYs, Ndim)
			for j := range xys {
				xys[j].X = float64(j)
				xys[j].Y = col.At(j, 0)
			}
			line, err := hplot.NewLine(xys)
			if err != nil {
				return err
			}
			line.Color = colors[i]
			p.Add(line)
			p.Legend.Add(fmt.Sprintf("PC%d", i), line)
		}
		return err
	})

	grp.Go(func() error {
		p := tp.Plot(1, 0)
		p.Title.Text = "Explained variance ratio"
		p.X.Label.Text = "j-th principal component"
		p.Y.Label.Text = "sigma_j / sigma_tot"
		p.X.Tick.Marker = &plot.LogTicks{}
		p.X.Scale = &plot.LogScale{}
		p.Y.Tick.Marker = &plot.LogTicks{}
		p.Y.Scale = &plot.LogScale{}
		vars := pca.VarsTo(nil)
		invSum := 1 / floats.Sum(vars)
		xys := make(plotter.XYs, len(vars))
		for i, v := range vars {
			xys[i].X = float64(i + 1)
			xys[i].Y = v * invSum
		}
		line, err := hplot.NewLine(xys)
		if err != nil {
			return err
		}
		line.Color = colors[2]
		p.Add(line)
		return nil
	})

	grp.Go(func() error {
		p := tp.Plot(1, 1)
		p.Title.Text = "Sample distribution"
		p.X.Label.Text = "Mean of samples"
		p.Y.Label.Text = "Probability density"
		//p.Y.Tick.Marker = &plot.LogTicks{}
		//p.Y.Scale = &plot.LogScale{}
		p.Y.Min = 1e-3
		p.Legend.Top = true
		for i, v := range []struct {
			name string
			data *mat.Dense
		}{
			{"1 sample", dataRaw},
			{"10^3 samples", meanInit},
			{"PCA [training]", meanDecorTrain},
			{"PCA [testing]", meanDecor},
		} {
			h := hbook.NewH1D(300, 450, 650)
			for _, vv := range v.data.RawMatrix().Data {
				h.Fill(vv, 1)
			}
			h.Scale(1 / h.Integral())

			hh := hplot.NewH1D(h)
			hh.Color = colors[i]
			hh.FillColor = nil
			p.Add(hh)
			p.Legend.Add(fmt.Sprintf("%s RMS=%.1f", v.name, h.XStdDev()), hh)
		}
		return nil
	})

	err = grp.Wait()
	if err != nil {
		log.Fatal(err)
	}

	err = tp.Save(-1, 20*vg.Centimeter, "results.png")
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("==> done (%v)", time.Since(t5))

	t6 := time.Now()
	log.Printf("creating correlation plots...")
	tp = hplot.NewTiledPlot(draw.Tiles{Rows: 2, Cols: 2})

	grp.Go(func() error {
		p := tp.Plot(0, 0)
		p.Title.Text = "Correlation matrix: before"
		p.X.Label.Text = "j-th sample"
		p.Y.Label.Text = "i-th sample"
		cov := stat.CorrelationMatrix(nil, dataRaw, nil)
		r, c := cov.Dims()
		h2d := hbook.NewH2D(50, 0, float64(c), 50, 0, float64(r))
		for ix := 0; ix < c; ix++ {
			x := float64(ix)
			for iy := 0; iy < r; iy++ {
				y := float64(iy)
				h2d.Fill(x, y, cov.At(ix, iy))
			}
		}
		p.Add(hplot.NewH2D(h2d, nil))
		return nil
	})

	grp.Go(func() error {
		p := tp.Plot(0, 1)
		p.Title.Text = "Correlation matrix: after"
		p.X.Label.Text = "j-th sample"
		p.Y.Label.Text = "i-th sample"
		cov := stat.CorrelationMatrix(nil, &dataDecor, nil)
		r, c := cov.Dims()
		h2d := hbook.NewH2D(50, 0, float64(c), 50, 0, float64(r))
		for ix := 0; ix < c; ix++ {
			x := float64(ix)
			for iy := 0; iy < r; iy++ {
				y := float64(iy)
				h2d.Fill(x, y, cov.At(ix, iy))
			}
		}
		p.Add(hplot.NewH2D(h2d, nil))
		return nil
	})

	grp.Go(func() error {
		p := tp.Plot(1, 0)
		p.Title.Text = "Scatter plot: before"
		p.X.Label.Text = "50-th PC"
		p.Y.Label.Text = "100-th PC"
		p.X.Min = -150
		p.X.Max = +150
		p.Y.Min = -150
		p.Y.Max = +150

		d1 := dataRaw.ColView(50)
		mean1 := mat.Sum(d1) / float64(rows)
		d2 := dataRaw.ColView(100)
		mean2 := mat.Sum(d2) / float64(rows)

		xys := make(plotter.XYs, rows)
		for i := range xys {
			xys[i].X = d1.At(i, 0) - mean1
			xys[i].Y = d2.At(i, 0) - mean2
		}
		pts, err := plotter.NewScatter(xys)
		if err != nil {
			return err
		}
		pts.Color = colors[2]
		pts.Radius = 0.2 * vg.Millimeter
		p.Add(pts)
		return nil
	})

	grp.Go(func() error {
		p := tp.Plot(1, 1)
		p.Title.Text = "Scatter plot: after"
		p.X.Label.Text = "50-th sample"
		p.Y.Label.Text = "100-th sample"
		p.X.Min = -150
		p.X.Max = +150
		p.Y.Min = -150
		p.Y.Max = +150

		rows, _ := dataDecor.Dims()
		d1 := dataDecor.ColView(50)
		mean1 := mat.Sum(d1) / float64(rows)
		d2 := dataDecor.ColView(100)
		mean2 := mat.Sum(d2) / float64(rows)

		xys := make(plotter.XYs, rows)
		for i := range xys {
			xys[i].X = d1.At(i, 0) - mean1
			xys[i].Y = d2.At(i, 0) - mean2
		}
		pts, err := plotter.NewScatter(xys)
		if err != nil {
			return err
		}
		pts.Color = colors[2]
		pts.Radius = 0.2 * vg.Millimeter
		p.Add(pts)

		return nil
	})

	err = grp.Wait()
	if err != nil {
		log.Fatal(err)
	}

	err = tp.Save(-1, 20*vg.Centimeter, "correlations.png")
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("==> done (%v)", time.Since(t6))
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

func init() {
	plotter.DefaultLineStyle.Width = vg.Points(2)
}
