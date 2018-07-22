package datasets

import (
	"fmt"
	"github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/score"
)

var (
	Digits Data
)

func init() {
	fmt.Println("Downloading DigitsX dataset...")
	scope, rawDataset := data.DigitsX()
	_, _, classVar, labels := data.ExtractLabels(scope, rawDataset)
	Digits = &dataProto{rawDataset, scope, labels, classVar, 20, 30, 8, score.NewScore()}
}
