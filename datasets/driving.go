package datasets

import (
	"fmt"
	"github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/io"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/score"
	"github.com/RenatoGeh/gospn/spn"
)

var (
	Driving Data
)

func pullData(trainFile, testFile string, n, m int) (spn.Dataset, []int, spn.Dataset, []int) {
	r, err := io.NewNpyReader(trainFile)
	if err != nil {
		panic(err)
	}
	t, err := io.NewNpyReader(testFile)
	if err != nil {
		panic(err)
	}
	defer r.Close()
	defer t.Close()

	D, L, err := r.ReadBalanced(n, 3)
	if err != nil {
		panic(err)
	}
	E, U, err := t.ReadBalanced(m, 3)
	if err != nil {
		panic(err)
	}
	return D, L, E, U
}

func prepare(n int, m int) (spn.Dataset, []int, spn.Dataset, []int, map[int]*learn.Variable) {
	k := 80 * 45
	S := make(map[int]*learn.Variable)
	for i := 0; i < k; i++ {
		S[i] = &learn.Variable{i, 256, ""}
	}
	S[k] = &learn.Variable{k, 3, "cmd"}
	const prefix = "/home/renatogeh/Documents/Research/datasets/self_driving/transformed/"
	D, L, E, U := pullData(prefix+"train.npy", prefix+"test.npy", n, m)
	return D, L, E, U, S
}

func init() {
	fmt.Println("Driving dataset...")
	R, L, _, _, Sc := prepare(600, 1)
	rawR := data.MergeLabel(R, L, Sc[3600])
	Driving = &dataProto{rawR, Sc, L, Sc[3600], 80, 45, 256, score.NewScore()}
}
