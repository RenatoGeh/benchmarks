package common

import (
	"fmt"
	"github.com/RenatoGeh/gospn/io"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/sys"
)

func subtract(U spn.VarSet, V spn.VarSet) {
	for k, _ := range V {
		delete(U, k)
	}
}

func Complete(S spn.SPN, D spn.Dataset, L []int, cv *learn.Variable, p float64, a string) {
	st := spn.NewStorer()
	tk := st.NewTicket()
	v := cv.Varid
	H := make(map[io.CmplType]spn.VarSet)
	for i, I := range D {
		//left, _ := io.SplitHalf(I, io.Left, sys.Width, sys.Height)
		//_, _, M := spn.StoreMAP(S, left, tk, st)
		//subtract(M, left)
		//io.ImgCmplToPGM(fmt.Sprintf("%d.pgm", int(p)), left, M, io.Left, sys.Width, sys.Height, sys.Max)
		//st.Reset(tk)
		delete(I, v)
		H[io.Left], H[io.Right] = io.SplitHalf(I, io.Left, sys.Width, sys.Height)
		H[io.Top], H[io.Bottom] = io.SplitHalf(I, io.Top, sys.Width, sys.Height)
		for t, h := range H {
			_, _, M := spn.StoreMAP(S, h, tk, st)
			s := fmt.Sprintf("cmpl_%s_%.1f_%d_%s_pl_%dx%d.pgm", t, p, i, a, M[v], L[i])
			subtract(M, h)
			delete(M, v)
			io.ImgCmplToPGM(s, h, M, t, sys.Width, sys.Height, sys.Max)
			st.Reset(tk)
			h[v] = L[i]
			_, _, M = spn.StoreMAP(S, h, tk, st)
			s = fmt.Sprintf("cmpl_%s_%.1f_%d_%s_nl.pgm", t, p, i, a)
			subtract(M, h)
			delete(M, v)
			io.ImgCmplToPGM(s, h, M, t, sys.Width, sys.Height, sys.Max)
			st.Reset(tk)
		}
		I[v] = L[i]
	}
}
