package mnist

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
	rawTest     spn.Dataset
	test        spn.Dataset
	testLabels  []int
	rawTrain    spn.Dataset
	train       spn.Dataset
	trainLabels []int
	scope       map[int]*learn.Variable
	classVar    *learn.Variable
)

func init() {
	fmt.Println("Downloading MNIST-3Bits-2000 dataset...")
	scope, rawTest, rawTrain = data.MNIST3Bits2000()
	_, test, classVar, testLabels = data.ExtractLabels(scope, rawTest)
	_, train, classVar, trainLabels = data.ExtractLabels(scope, rawTrain)
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
				//parameters.Bind(S, P.P)
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
				S := dennis.Structure(K[id], Sc, P.ClustersPerDecomp, P.SumsPerRegion, P.GaussPerPixel,
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

func Classify(A params.Algorithm) (*score.S, error) {
	sys.Width, sys.Height = 28, 28
	sys.Max = 8

	score := score.NewScore()
	st := spn.NewStorer()
	tk := st.NewTicket()
	t := A.Name()
	var S spn.SPN
	if t == "gens" {
		P := A.(*params.Gens)
		S = gens.Learn(scope, rawTrain, P.Clusters, P.PValue, P.Epsilon, P.MinPoints)
	} else {
		S = createStructure(A, rawTrain, scope, trainLabels, t)
	}
	fmt.Println("Evaluating scores...")
	n := len(test) / 10
	for i, I := range test {
		if i > 0 && i%n == 0 {
			fmt.Printf("... %d%% ...\n", int(100.0*(float64(i)/float64(len(test)))))
		}
		j := classVar.Varid
		delete(I, j)
		_, _, M := spn.StoreMAP(S, I, tk, st)
		score.Register(M[j], testLabels[i])
		st.Reset(tk)
	}
	return score, nil
}

func ClassifyIn(A params.Algorithm, p float64) (*score.S, error) {
	sys.Width, sys.Height = 28, 28
	sys.Max = 8

	allTrain, allLabels := data.Join(rawTrain, rawTest, trainLabels, testLabels)
	D, L := data.PartitionByLabels(allTrain, allLabels, classVar.Categories, []float64{p, 1.0 - p})
	T, R := D[0], D[1]
	tL, rL := L[0], L[1]
	score := score.NewScore()
	st := spn.NewStorer()
	tk := st.NewTicket()
	t := A.Name()
	var S spn.SPN
	if t == "gens" {
		P := A.(*params.Gens)
		S = gens.Learn(scope, R, P.Clusters, P.PValue, P.Epsilon, P.MinPoints)
	} else {
		S = createStructure(A, R, scope, rL, t)
	}
	fmt.Println("Evaluating scores...")
	n := len(T) / 10
	for i, I := range T {
		if i > 0 && i%n == 0 {
			fmt.Printf("... %d%% ...\n", int(100.0*(float64(i)/float64(len(T)))))
		}
		j := classVar.Varid
		delete(I, j)
		_, _, M := spn.StoreMAP(S, I, tk, st)
		score.Register(M[j], tL[i])
		st.Reset(tk)
	}
	return score, nil
}
