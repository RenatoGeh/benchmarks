package datasets

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
	"sync"
)

var (
	mu = &sync.Mutex{}
)

type dataProto struct {
	rawDataset spn.Dataset
	scope      map[int]*learn.Variable
	labels     []int
	classVar   *learn.Variable
	w, h, m    int
	scr        *score.S
}

type Data interface {
	Classify(A params.Algorithm, p float64) (*score.S, error)
	Complete(A params.Algorithm, p float64) error
	CompletePerLabel(A params.Algorithm)
	Dataset() (spn.Dataset, []int)
	ClassVar() *learn.Variable
}

func (d *dataProto) Dataset() (spn.Dataset, []int) { return d.rawDataset, d.labels }
func (d *dataProto) ClassVar() *learn.Variable     { return d.classVar }

func (d *dataProto) Classify(A params.Algorithm, p float64) (*score.S, error) {
	if p == 0.0 || p == 1.0 {
		return nil, errors.New("Partition value p can't be 0 or 1!")
	}
	mu.Lock()
	sys.Width, sys.Height = d.w, d.h
	sys.Max = d.m
	D, L := data.PartitionByLabels(d.rawDataset, d.labels, d.classVar.Categories, []float64{p, 1.0 - p})
	mu.Unlock()
	sys.Println("Creating structure...")
	S := common.ClassStructure(A, D[0], d.scope, L[0], d.classVar)
	sys.Println("Structure created.")
	d.scr.Clear()
	d.scr.Evaluate(D[1], L[1], S, d.classVar)
	return d.scr, nil
}

func (d *dataProto) Complete(A params.Algorithm, p float64) error {
	if p == 0.0 || p == 1.0 {
		return errors.New("Partition value p can't be 0 or 1!")
	}
	sys.Width, sys.Height = d.w, d.h
	sys.Max = d.m

	//for i := 0; i < len(d.rawDataset); i++ {
	//fmt.Printf("Instance %d\n", i)
	//I := d.rawDataset[i]
	//D := append(d.rawDataset[0:i], d.rawDataset[i+1:]...)
	//S := common.CmplStructure(A, D, d.scope, d.classVar)
	//common.Complete(S, []map[int]int{I}, nil, d.classVar, float64(i), "")
	//}

	D, L := data.PartitionByLabels(d.rawDataset, d.labels, d.classVar.Categories, []float64{p, 1.0 - p})
	fmt.Println("Generating structure...")
	S := common.CmplStructure(A, D[0], d.scope, d.classVar)
	fmt.Println("Completing...")
	common.Complete(S, D[1], L[1], d.classVar, p, A.Name())
	return nil
}

func (d *dataProto) CompletePerLabel(A params.Algorithm) {
	sys.Width, sys.Height, sys.Max = d.w, d.h, d.m
	n := d.classVar.Categories
	for i := 0; i < n; i++ {
		D, L, E, M := data.SubtractLabel(d.rawDataset, d.labels, i)
		S := common.ClassStructure(A, D, d.scope, L, d.classVar)
		fmt.Println("Created structure. Performing completion...")
		common.Complete(S, E, M, d.classVar, 1, "np")
		S = nil
		sys.ForceFree()
		fmt.Println("Completion done.")
		m := len(E)
		for j := 0; j < m; j++ {
			nE, nM := data.Copy(E, M)
			I := nE[j]
			nE = append(nE[:j], nE[j+1:]...)
			nM = append(nM[:j], nM[j+1:]...)
			nD, nL := data.Join(D, nE, L, nM)
			S = common.ClassStructure(A, nD, d.scope, nL, d.classVar)
			fmt.Println("Created structure. Performing completion...")
			common.Complete(S, []map[int]int{I}, []int{i}, d.classVar, 1.0, fmt.Sprintf("pf_%d", j))
			fmt.Println("Completion done.")
			S = nil
			nE, nM = nil, nil
			nD, nL = nil, nil
			sys.ForceFree()
		}
		D, L, E, M = nil, nil, nil, nil
		sys.ForceFree()
	}
}
