package dataset

import (
	"math"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

// getRowsAndCols is a helper that returns nRows and nCols that we'll divide the
// output image into
func (d *Dataset) getRowsAndCols() (int, int) {
	nDatasetCols := float64(len(d.Rows[0]))
	cols := int(math.Ceil(math.Sqrt(nDatasetCols)))
	rows := int(math.Ceil(nDatasetCols / float64(cols)))
	return rows, cols
}

// VisualizeColumnVsTarget creates one scatter plot for each non-target column and
// outputs it to outpath
func (d *Dataset) VisualizeColumnVsTarget(outpath string, width, height int) error {
	rows, cols := d.getRowsAndCols()

	names := []string{}
	vals := make([]plotter.XYs, len(d.Rows[0]))

	for r, row := range d.Rows {
		for c, name := range d.ColumnNames {
			names = append(names, name)
			if r == 0 {
				vals[c] = make(plotter.XYs, 0)
			}
			vals[c] = append(vals[c], plotter.XY{X: row[c], Y: row.Y()})
		}
	}

	plots := make([][]*plot.Plot, rows)
	for j := 0; j < rows; j++ {
		plots[j] = make([]*plot.Plot, cols)
		for i := 0; i < cols; i++ {
			pltIdx := (j * rows) + i
			// there might be some empty spaces, so we won't have data to put there
			if pltIdx < len(d.Rows[0]) {
				plt := plot.New()
				plt.Title.Text = names[pltIdx]
				scatt, err := plotter.NewScatter(vals[pltIdx])
				if err != nil {
					return err
				}
				plt.Add(scatt)

				plots[j][i] = plt
			}
		}
	}

	img := vgimg.New(vg.Points(float64(width)), vg.Points(float64(height)))
	dc := draw.New(img)

	t := draw.Tiles{
		Rows: rows,
		Cols: cols,
	}

	canvases := plot.Align(plots, t, dc)
	for j := 0; j < rows; j++ {
		for i := 0; i < cols; i++ {
			if plots[j][i] != nil {
				plots[j][i].Draw(canvases[j][i])
			}
		}
	}

	w, err := os.Create(outpath)
	if err != nil {
		panic(err)
	}

	png := vgimg.PngCanvas{Canvas: img}
	if _, err := png.WriteTo(w); err != nil {
		return err
	}

	return nil
}

// VisualizeColumns outputs an image to outpath containing histograms for each of
// the columns
func (d *Dataset) VisualizeColumnHistograms(outpath string, width, height int) error {
	rows, cols := d.getRowsAndCols()

	names := []string{}
	vals := make([]plotter.Values, len(d.Rows[0]))

	for r, row := range d.Rows {
		for c, name := range d.ColumnNames {
			names = append(names, name)
			if r == 0 {
				vals[c] = make(plotter.Values, 0)
			}
			vals[c] = append(vals[c], row[c])
		}
	}

	plots := make([][]*plot.Plot, rows)
	for j := 0; j < rows; j++ {
		plots[j] = make([]*plot.Plot, cols)
		for i := 0; i < cols; i++ {
			pltIdx := (j * rows) + i
			// there might be some empty spaces, so we won't have data to put there
			if pltIdx < len(d.Rows[0]) {
				plt := plot.New()
				plt.Title.Text = names[pltIdx]
				hist, err := plotter.NewHist(vals[pltIdx], 20)
				if err != nil {
					return err
				}
				plt.Add(hist)

				plots[j][i] = plt
			}
		}
	}

	img := vgimg.New(vg.Points(float64(width)), vg.Points(float64(height)))
	dc := draw.New(img)

	t := draw.Tiles{
		Rows: rows,
		Cols: cols,
	}

	canvases := plot.Align(plots, t, dc)
	for j := 0; j < rows; j++ {
		for i := 0; i < cols; i++ {
			if plots[j][i] != nil {
				plots[j][i].Draw(canvases[j][i])
			}
		}
	}

	w, err := os.Create(outpath)
	if err != nil {
		panic(err)
	}

	png := vgimg.PngCanvas{Canvas: img}
	if _, err := png.WriteTo(w); err != nil {
		return err
	}

	return nil
}
