package common

import (
	"errors"
	"fmt"
	"github.com/RenatoGeh/benchmarks/params"
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

func ClassStructure(A params.Algorithm, T spn.Dataset, Sc map[int]*learn.Variable, L []int, classVar *learn.Variable) spn.SPN {
	Q := conc.NewSingleQueue(-1)
	mu := &sync.Mutex{}
	if t := A.Name(); t == "gens" {
		P := A.(*params.Gens)
		return gens.Learn(Sc, T, P.Clusters, P.PValue, P.Epsilon, P.MinPoints)
	} else if t == "dennis" {
		P := A.(*params.Dennis)
		c := classVar.Categories
		K := data.Split(T, c, L)
		root := spn.NewSum()
		for i := 0; i < c; i++ {
			if len(K[i]) <= 0 {
				continue
			}
			Q.Run(func(id int) {
				mu.Lock()
				lsc := make(map[int]*learn.Variable)
				for k, v := range Sc {
					if k != classVar.Varid {
						lsc[k] = v
					}
				}
				mu.Unlock()
				sys.Printf("Creating structure for class %d...\n", id)
				S := dennis.Structure(K[id], lsc, P.ClustersPerDecomp, P.SumsPerRegion, P.GaussPerPixel,
					P.SimilarityThreshold)
				parameters.Bind(S, P.P)
				sys.Println("Generative learning...")
				learn.Generative(S, K[id])
				pi := spn.NewProduct()
				pi.AddChild(S)
				pi.AddChild(spn.NewIndicator(classVar.Varid, id))
				mu.Lock()
				root.AddChildW(pi, 1.0/float64(c))
				mu.Unlock()
				sys.Printf("Created structure for class %d.\n", id)
			}, i)
		}
		Q.Wait()
		return root
	} else if t == "poon" {
		P := A.(*params.Poon)
		c := classVar.Categories
		K := data.Split(T, c, L)
		root := spn.NewSum()
		for i := 0; i < c; i++ {
			if len(K[i]) <= 0 {
				continue
			}
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

func CmplStructure(A params.Algorithm, T spn.Dataset, Sc map[int]*learn.Variable, classVar *learn.Variable) spn.SPN {
	if t := A.Name(); t == "gens" {
		P := A.(*params.Gens)
		return gens.Learn(Sc, T, P.Clusters, P.PValue, P.Epsilon, P.MinPoints)
	} else if t == "dennis" {
		P := A.(*params.Dennis)
		S := dennis.Structure(T, Sc, P.ClustersPerDecomp, P.SumsPerRegion, P.GaussPerPixel,
			P.SimilarityThreshold)
		parameters.Bind(S, P.P)
		fmt.Println("Generative learning...")
		learn.Generative(S, T)
		return S
	} else if t == "poon" {
		P := A.(*params.Poon)
		R := data.SubtractVariable(T, classVar)
		S := poon.Structure(R, P.SumsPerRegion, P.GaussPerPixel, P.Resolution)
		parameters.Bind(S, P.P)
		fmt.Println("Generative learning...")
		learn.Generative(S, R)
		return S
	} else {
		panic(errors.New(fmt.Sprintf("Unrecognized algorithm: %s", t)))
	}
}
