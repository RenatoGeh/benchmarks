package main

import (
	"fmt"
	"github.com/RenatoGeh/benchmarks/datasets"
	"github.com/RenatoGeh/benchmarks/params"
	"github.com/RenatoGeh/gospn/learn/parameters"
	"github.com/RenatoGeh/gospn/sys"
	"os"
)

func testDigits(A params.Algorithm) {
	for p := 0.1; p < 0.95; p += 0.1 {
		score, err := datasets.Digits.Classify(A, p)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		fmt.Printf("At p=%.1f:\n", p)
		fmt.Println(score)
	}
}

func testSepMNIST(A params.Algorithm) {
	score, err := datasets.MNIST.ClassifySep(A)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	fmt.Println(score)
}

func testMNIST(A params.Algorithm) {
	for p := 0.1; p < 0.95; p += 0.1 {
		score, err := datasets.MNIST.Classify(A, p)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		fmt.Printf("At p=%.1f:\n", p)
		fmt.Println(score)
	}
}

func testCmplDigits(A params.Algorithm) {
	//for p := 0.1; p < 0.95; p += 0.1 {
	if err := datasets.Digits.Complete(A, 0.7); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	//}
}

func testCmplOlivetti(A params.Algorithm) {
	//datasets.Olivetti.CompletePerLabel(A)
	if err := datasets.Olivetti.Complete(A, 0.7); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func main() {
	sys.Verbose = false
	P := parameters.New(true, false, 0.01, parameters.HardGD, 0.01, 1.0, 0, 0.1, 4)
	//A := params.NewDennis(P, 4, 4, 1, 0.95)
	//A := params.NewPoon(P, 4, 4, 2)
	A := params.NewGens(P, -1, 0.01, 4, 4)
	//testCmplDigits(A)
	//testDigits(A)
	testCmplOlivetti(A)
	//testMNIST(A)
}
