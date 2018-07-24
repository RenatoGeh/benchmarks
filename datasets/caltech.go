package datasets

import (
	"fmt"
	"github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/score"
)

var (
	Caltech Data
)

func init() {
	fmt.Println("Downloading Caltech-101 dataset...")
	scope, rawDataset := data.Caltech3Bit()
	_, _, classVar, labels := data.ExtractLabels(scope, rawDataset)
	Caltech = &dataProto{rawDataset, scope, labels, classVar, 150, 65, 16, score.NewScore()}
}
