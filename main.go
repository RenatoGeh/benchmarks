package main

import (
	"fmt"
	"github.com/RenatoGeh/benchmarks/datasets"
	"github.com/RenatoGeh/benchmarks/params"
	"github.com/RenatoGeh/gospn/conc"
	"github.com/RenatoGeh/gospn/learn/parameters"
	"github.com/RenatoGeh/gospn/score"
	"github.com/RenatoGeh/gospn/sys"
	"os"
	"sync"
)

func testSingle(A params.Algorithm, p float64, D datasets.Data) *score.S {
	sc, err := D.Classify(A, p)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	return sc
}

func testDigits(A params.Algorithm, p float64) *score.S {
	sc, err := datasets.Digits.Classify(A, p)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	//fmt.Printf("At p=%.1f:\n", p)
	//fmt.Println(sc)
	return sc
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
	//for p := 0.6; p < 0.95; p += 0.1 {
	//fmt.Printf("Starting classification with p=%.1f...\n", p)
	//score, err := datasets.MNIST.Classify(A, p)
	//if err != nil {
	//fmt.Println(err)
	//os.Exit(1)
	//}
	//fmt.Printf("At p=%.1f:\n", p)
	//fmt.Println(score)
	//}
	mu := &sync.Mutex{}
	Q := conc.NewSingleQueue(-1)
	var P []float64
	for p := 0.2; p < 0.95; p += 0.1 {
		P = append(P, p)
	}
	for i := 0; i < 9; i++ {
		Q.Run(func(id int) {
			score, err := datasets.MNIST.Classify(A, P[id])
			if err != nil {
				fmt.Println(err)
				os.Exit(1)
			}
			mu.Lock()
			fmt.Printf("At p=%.1f:\n", P[id])
			fmt.Println(score)
			mu.Unlock()
		}, i)
	}
	Q.Wait()
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

func testCaltech(A params.Algorithm) {
	mu := &sync.Mutex{}
	Q := conc.NewSingleQueue(-1)
	var P []float64
	for p := 0.1; p < 0.95; p += 0.1 {
		P = append(P, p)
	}
	for i := 0; i < 9; i++ {
		Q.Run(func(id int) {
			score, err := datasets.Caltech.Classify(A, P[id])
			if err != nil {
				fmt.Println(err)
				os.Exit(1)
			}
			mu.Lock()
			fmt.Printf("At p=%.1f:\n", P[id])
			fmt.Println(score)
			mu.Unlock()
		}, i)
	}
	Q.Wait()
}

func testDriving(A params.Algorithm) {
	mu := &sync.Mutex{}
	Q := conc.NewSingleQueue(2)
	var P []float64
	for p := 0.1; p < 0.95; p += 0.1 {
		P = append(P, p)
	}
	for i := 0; i < 9; i++ {
		Q.Run(func(id int) {
			score, err := datasets.Driving.Classify(A, P[id])
			if err != nil {
				fmt.Println(err)
				os.Exit(1)
			}
			mu.Lock()
			fmt.Printf("At p=%.1f:\n", P[id])
			fmt.Println(score)
			mu.Unlock()
		}, i)
	}
	Q.Wait()
}

func main() {
	sys.Verbose = false
	P := parameters.New(true, false, 0.01, parameters.HardGD, 0.01, 1.0, 0, 0.1, 4)
	//A := params.NewDennis(P, 4, 4, 1, 0.95)
	//A := params.NewPoon(P, 4, 4, 4)
	A := params.NewGens(P, -1, 0.1, 4, 4)
	//testDriving(A)
	testMNIST(A)
	return
	//testCaltech(A)
	//testCmplDigits(A)
	//testCmplOlivetti(A)
	//testSepMNIST(A)
	n := 5
	Q := conc.NewSingleQueue(-1)
	mu := &sync.Mutex{}
	for p := 0.5; p < 0.95; p += 0.1 {
		S := score.NewScore()
		for i := 0; i < n; i++ {
			Q.Run(func(id int) {
				s := testSingle(A, p, datasets.Caltech)
				mu.Lock()
				S.Merge(s)
				mu.Unlock()
			}, i)
		}
		Q.Wait()
		fmt.Println("=======================================")
		fmt.Printf("Average of %d iterations (p=%.1f):\n", n, p)
		fmt.Println(S)
		fmt.Println("=======================================")
	}
}
