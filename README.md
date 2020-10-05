# OptiMap - Optical-map molecule alignment tool

## Requirements:

Depends on [OptiScan](https://gitlab.com/akdel/OptiScan)

## Installation

```shell script
git clone https://gitlab.com/akdel/OptiMap.git
cd OptiMap
pip install .
```

## Test with toy data
```shell script
OptiSpeed ./data/molecules.bnx --first_score=0.8 --second_score=0.75 --output_filename=test_output.tsv --combine_sparse=True
OptiSpeed-denoise ./data/molecules.npy --first_score=0.8 --second_score=0.75 --output_filename=denoised_molecules.npy
OptiSpeed ./data/denoised_molecules.bnx --first_score=0.8 --second_score=0.75 --output_filename=test_output.tsv --combine_sparse=True
```