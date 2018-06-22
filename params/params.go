package params

import (
	"github.com/RenatoGeh/gospn/learn/parameters"
)

type Algorithm interface {
	Name() string
	LearningParams() *parameters.P
}

type baseAlgorithm struct {
	P *parameters.P
}

func (a *baseAlgorithm) LearningParams() *parameters.P { return a.P }

// Poon-Domingos
type Poon struct {
	baseAlgorithm
	SumsPerRegion int
	GaussPerPixel int
	Resolution    int
}

func NewPoon(P *parameters.P, m, g, r int) *Poon {
	return &Poon{baseAlgorithm{P}, m, g, r}
}

func (p *Poon) Name() string { return "poon" }

// Dennis-Ventura
type Dennis struct {
	baseAlgorithm
	SumsPerRegion       int
	GaussPerPixel       int
	ClustersPerDecomp   int
	SimilarityThreshold float64
}

func NewDennis(P *parameters.P, m, g, k int, t float64) *Dennis {
	return &Dennis{baseAlgorithm{P}, m, g, k, t}
}

func (d *Dennis) Name() string { return "dennis" }

// Gens-Domingos
type Gens struct {
	baseAlgorithm
	// If Clusters <= 0, use DBSCAN.
	Clusters int
	PValue   float64
	// DBSCAN only.
	Epsilon   float64
	MinPoints int
}

func NewGens(P *parameters.P, k int, p, e float64, m int) *Gens {
	return &Gens{baseAlgorithm{P}, k, p, e, m}
}

func (g *Gens) Name() string { return "gens" }
