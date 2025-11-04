# Benchmarks of the Leopard-EM package

Identifying freely oriented macromolecules using Two-Dimensional Template Matching (2DTM) is a computationally intensive task since we must compute millions of cross-correlograms on large cryo-EM micrographs.
Efficiency of the `match_template` program is therefore a key consideration going into the Leopard-EM package.

We include some benchmarking results across different GPU hardware to provide an estimate of `match_template` performance.
These results can help guide users in planning out 2DTM analyses of their data using Leopard-EM and serve as a reference for expected performance.

## Benchmarking Setup

Leopard-EM includes a benchmarking script at `benchmark/benchmark_match_template.py` (if downloaded from source) which you can use to determine performance _on your own hardware_.
This script runs the `match_template` program using the following parameters:

- Micrograph size: 4096 x 4096 pixels (Falcon 4i) or 5760 x 4092 pixels (K3)
- Template size: 512 x 512 x 512 pixels
- Number of defocus planes: 11
- Variable orientation batch size configurable using `--orientation-batch-size`

Note that we empirically observe that template size has negligible effect on performance.
Total search times are extrapolated from throughput to a full orientation search of ~1.58 million orientations with 13 defocus planes (~20.5 million total cross-correlations).

## Version 1.1 benchmarks

### Falcon 4i images (4096 x 4096 pixels)

GPU name                 | VRAM  | Image size | Throughput (cross-corr/sec) | 2DTM search time (hours) |
------------------------ | ----- | ---------- | --------------------------- | -------------------------- |
GeForce 2080 Ti          | 11 GB | 4096×4096  | 343.0                       | 16.70                      |
RTX 6000 Ada / L40s      | 48 GB | 4096×4096  | 744.5                       | 7.69                       |
RTX 6000 Blackwell Max-Q | 96 GB | 4096×4096  | 1394.7                      | 4.10                       |
A100                     | 40 GB | 4096x4096  | 923.4                       | 6.19                       |
H100                     | 80 GB | 4096×4096  | 1650.8                      | 3.47                       |

### K3 images (5760 x 4092 pixels)

GPU name                 | VRAM  | Image size | Throughput (cross-corr/sec) | 2DTM search time (hours) |
------------------------ | ----- | ---------- | --------------------------- | -------------------------- |
GeForce 2080 Ti          | 11 GB | 5760×4092  | 217.1                       | 26.40                      |
RTX 6000 Ada / L40s      | 48 GB | 5760×4092  | 431.7                       | 13.30                      |
RTX 6000 Blackwell Max-Q | 96 GB | 5760×4092  | 799.7                       | 7.15                       |
A100                     | 40 GB | 5760×4092  | 530.2                       | 10.79                      |
H100                     | 80 GB | 5760×4092  | 897.9                       | 6.37                       |

!!! note "K3 image benchmarks"
    Note that we have not optimized Leopard-EM v1.1 for K3 images in particular. Future versions should include optimizations for non-square images which will improve performance on K3 data.