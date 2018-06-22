package score

import "fmt"

type S struct {
	Hits        int
	Misses      int
	Total       int
	Predictions []Pair
}

type Pair struct {
	Predicted int
	Expected  int
}

func NewScore() *S { return &S{} }

func (s *S) Register(predicted int, expected int) {
	if predicted == expected {
		s.Hits++
	} else {
		s.Misses++
	}
	s.Total++
	s.Predictions = append(s.Predictions, Pair{predicted, expected})
}

func (s *S) String() string {
	var str string
	str = fmt.Sprintf("Hits: %d\nMisses: %d\nTotal: %d\nAccuracy: %.5f\n", s.Hits, s.Misses, s.Total,
		float64(s.Hits)/float64(s.Total))
	str += fmt.Sprintf("Wrong predictions:\n")
	for _, p := range s.Predictions {
		if p.Predicted != p.Expected {
			str += fmt.Sprintf("  Expected %d, got %d.\n", p.Expected, p.Predicted)
		}
	}
	return str
}
