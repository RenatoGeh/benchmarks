package digits

import (
	"errors"
	"fmt"
	"github.com/RenatoGeh/benchmarks/params"
	"github.com/RenatoGeh/benchmarks/score"
	"github.com/RenatoGeh/gospn/conc"
	"github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/learn/dennis"
	"github.com/RenatoGeh/gospn/learn/gens"
	"github.com/RenatoGeh/gospn/learn/parameters"
	"github.com/RenatoGeh/gospn/learn/poon"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/sys"
	"sync"
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

func createStructure(A params.Algorithm, T spn.Dataset, Sc map[int]*learn.Variable, L []int, t string) spn.SPN {
	Q := conc.NewSingleQueue(-1)
	mu := &sync.Mutex{}
	if t == "poon" {
		P := A.(*params.Poon)
		c := classVar.Categories
		K := data.Split(T, c, L)
		root := spn.NewSum()
		for i := 0; i < c; i++ {
			Q.Run(func(id int) {
				mu.Lock()
				lsc := make(map[int]*learn.Variable)
				for k, v := range Sc {
					lsc[k] = v
				}
				mu.Unlock()
				fmt.Printf("Creating structure for digit %d...\n", id)
				S := poon.Structure(K[id], P.SumsPerRegion, P.GaussPerPixel, P.Resolution)
				parameters.Bind(S, P.P)
				//fmt.Println("Generative learning...")
				//learn.Generative(S, K[id])
				pi := spn.NewProduct()
				pi.AddChild(S)
				pi.AddChild(spn.NewIndicator(classVar.Varid, id))
				mu.Lock()
				root.AddChildW(pi, 1.0/float64(c))
				mu.Unlock()
				fmt.Printf("Created structure for digit %d.\n", id)
			}, i)
		}
		Q.Wait()
		return root
	} else if t == "dennis" {
		P := A.(*params.Dennis)
		c := classVar.Categories
		K := data.Split(T, c, L)
		root := spn.NewSum()
		for i := 0; i < c; i++ {
			Q.Run(func(id int) {
				mu.Lock()
				lsc := make(map[int]*learn.Variable)
				for k, v := range Sc {
					lsc[k] = v
				}
				mu.Unlock()
				fmt.Printf("Creating structure for digit %d...\n", id)
				S := dennis.Structure(K[id], lsc, P.ClustersPerDecomp, P.SumsPerRegion, P.GaussPerPixel,
					P.SimilarityThreshold)
				parameters.Bind(S, P.P)
				fmt.Println("Generative learning...")
				learn.Generative(S, K[id])
				pi := spn.NewProduct()
				pi.AddChild(S)
				pi.AddChild(spn.NewIndicator(classVar.Varid, id))
				mu.Lock()
				root.AddChildW(pi, 1.0/float64(c))
				mu.Unlock()
				fmt.Printf("Created structure for digit %d.\n", id)
			}, i)
		}
		Q.Wait()
		return root
	} else {
		panic(errors.New(fmt.Sprintf("Unrecognized algorithm: %s", t)))
	}
}

func Classify(A params.Algorithm, p float64) (*score.S, error) {
	if p == 0.0 || p == 1.0 {
		return nil, errors.New("Partition value p can't be 0 or 1!")
	}
	sys.Width, sys.Height = 20, 30
	sys.Max = 8

	D, L := data.PartitionByLabels(rawDataset, labels, classVar.Categories, []float64{p, 1.0 - p})
	score := score.NewScore()
	st := spn.NewStorer()
	tk := st.NewTicket()
	t := A.Name()
	var S spn.SPN
	if t == "gens" {
		P := A.(*params.Gens)
		S = gens.Learn(scope, D[0], P.Clusters, P.PValue, P.Epsilon, P.MinPoints)
	} else {
		S = createStructure(A, D[0], scope, L[0], t)
	}
	for i, I := range D[1] {
		j := classVar.Varid
		delete(I, j)
		_, _, M := spn.StoreMAP(S, I, tk, st)
		score.Register(M[j], L[1][i])
		st.Reset(tk)
	}
	return score, nil
}
