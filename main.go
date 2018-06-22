package main

import (
	"fmt"
	"github.com/RenatoGeh/benchmarks/digits"
	"github.com/RenatoGeh/benchmarks/params"
	"github.com/RenatoGeh/gospn/learn/parameters"
	"os"
)

func main() {
	//poon := params.NewPoon(parameters.Default(), 4, 4, 4)
	gens := params.NewGens(parameters.Default(), -1, 0.01, 4, 4)
	score, err := digits.Classify(gens, 0.7)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	fmt.Println(score)
}
