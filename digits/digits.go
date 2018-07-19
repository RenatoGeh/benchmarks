package digits

import (
	"errors"
	"fmt"
	"github.com/RenatoGeh/benchmarks/common"
	"github.com/RenatoGeh/benchmarks/params"
	"github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/score"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/sys"
)

var (
	rawDataset spn.Dataset
	scope      map[int]*learn.Variable
	labels     []int
	classVar   *learn.Variable
)

func init() {
	fmt.Println("Downloading DigitsX dataset...")
	scope, rawDataset = data.DigitsX()
	_, _, classVar, labels = data.ExtractLabels(scope, rawDataset)
}

func Classify(A params.Algorithm, p float64) (*score.S, error) {
	if p == 0.0 || p == 1.0 {
		return nil, errors.New("Partition value p can't be 0 or 1!")
	}
	sys.Width, sys.Height = 20, 30
	sys.Max = 8

	D, L := data.PartitionByLabels(rawDataset, labels, classVar.Categories, []float64{p, 1.0 - p})
	score := score.NewScore()
	S := common.ClassStructure(A, D[0], scope, L[0], classVar)
	score.Evaluate(D[1], L[1], S, classVar)
	return score, nil
}
