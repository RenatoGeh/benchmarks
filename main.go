package main

import (
	"fmt"
	"github.com/RenatoGeh/benchmarks/digits"
	"github.com/RenatoGeh/benchmarks/mnist"
	"github.com/RenatoGeh/benchmarks/params"
	"github.com/RenatoGeh/gospn/learn/parameters"
	"github.com/RenatoGeh/gospn/sys"
	"os"
)

func testDigits(A params.Algorithm) {
	for p := 0.1; p < 0.95; p += 0.1 {
		score, err := digits.Classify(A, p)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		fmt.Printf("At p=%.1f:\n", p)
		fmt.Println(score)
	}
}

func testMNIST(A params.Algorithm) {
	score, err := mnist.Classify(A)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	fmt.Println(score)
}

func testInMNIST(A params.Algorithm) {
	for p := 0.1; p < 0.95; p += 0.1 {
		score, err := digits.Classify(A, p)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		fmt.Printf("At p=%.1f:\n", p)
		fmt.Println(score)
	}
}

func main() {
	sys.Verbose = false
	P := parameters.New(true, false, 0.01, parameters.HardGD, 0.01, 1.0, 0, 0.1, 4)
	//A := params.NewDennis(P, 4, 4, 1, 0.95)
	A := params.NewPoon(P, 4, 4, 2)
	//A := params.NewGens(P, -1, 0.01, 4, 4)
	testDigits(A)
	//testMNIST(A)
}
