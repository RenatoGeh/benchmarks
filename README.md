# benchmarks

This repository contains code for easy benchmarking of different
SPN learning algorithms present in the
[GoSPN](https://github.com/RenatoGeh/gospn) library.

## Datasets

We used the following datasets to train the SPNs:

- [DigitsX](https://github.com/RenatoGeh/datasets/tree/master/digits_x)
- [Caltech-101 modified](https://github.com/RenatoGeh/datasets/tree/master/caltech_4bit)
- [Olivetti Faces modified](https://github.com/RenatoGeh/datasets/tree/master/olivetti_3bit)
- [MNIST subset](https://github.com/RenatoGeh/datasets/tree/master/mnist)

More information at https://github.com/RenatoGeh/datasets/.

## Results

We used the following parameters:

- Gens-Domingos: `pval=0.01`, `clusters=-1`, `epsilon=4`, `mp=4`
- Dennis-Ventura: `sumsPerRegion=4`, `gaussPerPixel=4`, `clustersPerDecomp=1`, `similarityThreshold=0.95`
- Poon-Domingos: `sumsPerRegion=4`, `gaussPerPixel=4`, `resolution=4`

When generative gradient descent was used, we set the following parameters:

* `Normalize=true`
* `HardWeight=false`
* `SmoothSum=0.01`
* `LearningType=parameters.HardGD`
* `Eta=0.01`
* `Epsilon=1.0`
* `BatchSize=0`
* `Lambda=0.1`
* `Iterations=4`

The Poon-Domingos algorithm either exceeded the time or memory limit, or
had unsatisfactory results. We'll look into that.

For each dataset, a percentage `p` of the dataset is set as training set and
`1-p` as test set. For MNIST, we used a fixed number of 2000 training
samples and 2000 test images, where no image in the training set had the
same handwritting as an image in the test set. We call an in-sample
result when we join the training and test set, shuffle the union and
then partition half of it as training and the rest as test. Out-sample
is when we simply take the two original training and test sets.

### Classification

All classification accuracy results are in percentage of hits.

#### DigitsX

Partition `p` | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9
--------------|-----|-----|-----|-----|-----|-----|-----|-----|----
Dennis-Ventura|92.85|98.57|99.18|98.81|99.42|99.28|98.57|93.33|88.75
Gens-Domingos |91.27|96.78|96.93|98.09|97.14|97.85|97.61|92.66|86.25

#### Caltech-101

Partition `p` | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9
--------------|-----|-----|-----|-----|-----|-----|-----|-----|----
Dennis-Ventura|78.58|78.49|80.28|79.88|81.38|81.35|75.45|74.78|75.75
Gens-Domingos |77.40|85.00|84.28|86.11|88.66|90.00|92.22|90.00|84.84

#### Olivetti Faces

Partition `p` | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9
--------------|-----|-----|-----|-----|-----|-----|-----|-----|----
Dennis-Ventura|83.78|74.88|89.85|89.93|96.22|97.50|92.89|50.00|60.93
Gens-Domingos | 2.50| 2.50|93.92|91.25|95.50|98.75|81.93|81.59|100.00

#### MNIST (2000 sample size)

Classifications|Dennis-Ventura|Gens-Domingos
---------------|--------------|-------------
In-sample|77.85|81.55
Out-sample|69.90|76.90
