package datasets

import (
	"fmt"
	"github.com/RenatoGeh/benchmarks/common"
	"github.com/RenatoGeh/benchmarks/params"
	"github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/score"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/sys"
)

type mnistData struct {
	dataProto
	rawTest     spn.Dataset
	rawTrain    spn.Dataset
	test        spn.Dataset
	train       spn.Dataset
	testLabels  []int
	trainLabels []int
}

var (
	MNIST *mnistData
)

func init() {
	fmt.Println("Downloading MNIST-3Bits-2000 dataset...")
	scope, rawTest, rawTrain := data.MNIST3Bits1000()
	_, test, classVar, testLabels := data.ExtractLabels(scope, rawTest)
	_, train, classVar, trainLabels := data.ExtractLabels(scope, rawTrain)
	allTrain, allLabels := data.Join(rawTrain, rawTest, trainLabels, testLabels)
	MNIST = &mnistData{dataProto{allTrain, scope, allLabels, classVar, 28, 28, 8, score.NewScore()},
		rawTest, rawTrain, test, train, testLabels, trainLabels}
}

func (d *mnistData) ClassifySep(A params.Algorithm) (*score.S, error) {
	sys.Width, sys.Height = d.w, d.h
	sys.Max = d.m
	S := common.ClassStructure(A, d.rawTrain, d.scope, d.trainLabels, d.classVar)
	d.scr.Evaluate(d.rawTest, d.testLabels, S, d.classVar)
	return d.scr, nil
}
