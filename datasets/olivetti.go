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
	scope, rawDataset := data.Olivetti3Bit()
	_, _, classVar, labels := data.ExtractLabels(scope, rawDataset)
	Olivetti = &dataProto{rawDataset, scope, labels, classVar, 46, 56, 8, score.NewScore()}
}
