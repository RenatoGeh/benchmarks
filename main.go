package main

import (
	"fmt"
	"github.com/RenatoGeh/benchmarks/digits"
	"github.com/RenatoGeh/benchmarks/params"
	"github.com/RenatoGeh/gospn/learn/parameters"
	"github.com/RenatoGeh/gospn/sys"
	"os"
)

func main() {
	sys.Verbose = false
	dennis := params.NewDennis(parameters.Default(), 4, 4, 1, 0.85)
	//poon := params.NewPoon(parameters.Default(), 4, 4, 2)
	//gens := params.NewGens(parameters.Default(), -1, 0.01, 4, 4)
	for p := 0.1; p < 0.95; p += 0.1 {
		score, err := digits.Classify(dennis, p)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		fmt.Printf("At p=%.1f:\n", p)
		fmt.Println(score)
		//score.Save(fmt.Sprintf("gens_score_p_%.1f.txt", p))
	}
}
