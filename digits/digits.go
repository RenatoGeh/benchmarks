package digits

import (
	"errors"
	"fmt"
	"github.com/RenatoGeh/benchmarks/params"
	"github.com/RenatoGeh/benchmarks/score"
	"github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/learn/dennis"
	"github.com/RenatoGeh/gospn/learn/gens"
	"github.com/RenatoGeh/gospn/learn/parameters"
	"github.com/RenatoGeh/gospn/learn/poon"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/sys"
)

var (
	raw_dataset spn.Dataset
	dataset     spn.Dataset
	scope       map[int]*learn.Variable
	labels      []int
	classVar    *learn.Variable
)

func init() {
	fmt.Println("Downloading DigitsX dataset...")
	scope, raw_dataset = data.DigitsX()
	_, dataset, classVar, labels = data.ExtractLabels(scope, raw_dataset)
}

func createStructure(A params.Algorithm, T spn.Dataset, Sc map[int]*learn.Variable, L []int, t string) spn.SPN {
	if t == "poon" {
		P := A.(*params.Poon)
		c := classVar.Categories
		K := data.Split(T, c, L)
		S := make([]spn.SPN, c)
		root := spn.NewSum()
		for i := 0; i < c; i++ {
			S[i] = poon.Structure(K[i], P.SumsPerRegion, P.GaussPerPixel, P.Resolution)
			pi := spn.NewProduct()
			root.AddChildW(pi, 1.0/float64(c))
			pi.AddChild(S[i])
			pi.AddChild(spn.NewIndicator(classVar.Varid, i))
		}
		parameters.Bind(root, P.P)
		return root
	} else if t == "dennis" {
		P := A.(*params.Dennis)
		c := classVar.Categories
		K := data.Split(T, c, L)
		S := make([]spn.SPN, c)
		root := spn.NewSum()
		for i := 0; i < c; i++ {
			S[i] = dennis.Structure(K[i], Sc, P.ClustersPerDecomp, P.SumsPerRegion, P.GaussPerPixel,
				P.SimilarityThreshold)
			pi := spn.NewProduct()
			root.AddChildW(pi, 1.0/float64(c))
			pi.AddChild(S[i])
			pi.AddChild(spn.NewIndicator(classVar.Varid, i))
		}
		parameters.Bind(root, P.P)
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

	D, L := data.PartitionByLabels(raw_dataset, labels, classVar.Categories, []float64{p, 1.0 - p})
	score := score.NewScore()
	st := spn.NewStorer()
	tk := st.NewTicket()
	t := A.Name()
	if t == "gens" {
		P := A.(*params.Gens)
		S := gens.Learn(scope, D[0], P.Clusters, P.PValue, P.Epsilon, P.MinPoints)
		for i, I := range D[1] {
			j := classVar.Varid
			delete(I, j)
			_, _, M := spn.StoreMAP(S, I, tk, st)
			score.Register(M[j], L[1][i])
			st.Reset(tk)
		}
		return score, nil
	}

	S := createStructure(A, D[0], scope, L[0], t)
	learn.Generative(S, D[0])
	for _, I := range D[1] {
		_, _, M := spn.StoreMAP(S, I, tk, st)
		j := classVar.Varid
		score.Register(M[j], L[1][j])
	}
	return score, nil
}
