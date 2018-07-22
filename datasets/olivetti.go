package datasets

import (
	"fmt"
	"github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/score"
)

var (
	Olivetti Data
)

func init() {
	fmt.Println("Downloading Olivetti dataset...")
	scope, rawDataset := data.OlivettiPadded()
	_, _, classVar, labels := data.ExtractLabels(scope, rawDataset)
	Olivetti = &dataProto{rawDataset, scope, labels, classVar, 48, 56, 256, score.NewScore()}
}
