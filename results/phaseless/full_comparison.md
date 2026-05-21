# Full Phaseless Strict Reproduction Comparison

- paper: `arXiv:2403.02584v2`
- device: `cuda`
- seed: `0`
- protocol: strict (Helmholtz forward + Eq. 3.11 DSM input, no proxy)

## Table 1 (Polygon accuracy)

| case | paper | ours | diff |
|---|---:|---:|---:|
| Ni=1,delta=0.02 | 0.9949 | 0.9904 | -0.0045 |
| Ni=1,delta=0.10 | 0.9772 | 0.9809 | +0.0037 |
| Ni=4,delta=0.02 | 0.9977 | 0.9932 | -0.0045 |
| Ni=4,delta=0.10 | 0.9916 | 0.9908 | -0.0008 |

## Table 2 (Relative L2)

| case | paper | ours | diff |
|---|---:|---:|---:|
| mnist,Ni=4,delta=0.05 | 0.0827 | 0.0989 | +0.0162 |
| mnist,Ni=4,delta=0.10 | 0.1043 | 0.1077 | +0.0034 |
| mnist,Ni=16,delta=0.05 | 0.0617 | 0.0797 | +0.0180 |
| mnist,Ni=16,delta=0.10 | 0.0755 | 0.0871 | +0.0116 |
| chinese_like,Ni=4,delta=0.05 | 0.1096 | 0.1387 | +0.0291 |
| chinese_like,Ni=4,delta=0.10 | 0.1252 | 0.1496 | +0.0244 |
| chinese_like,Ni=16,delta=0.05 | 0.0721 | 0.1254 | +0.0533 |
| chinese_like,Ni=16,delta=0.10 | 0.0854 | 0.1452 | +0.0598 |
| austria_ring_1,Ni=4,delta=0.05 | 0.1163 | 0.1083 | -0.0080 |
| austria_ring_1,Ni=4,delta=0.10 | 0.1258 | 0.1290 | +0.0032 |
| austria_ring_1,Ni=16,delta=0.05 | 0.0851 | 0.0882 | +0.0031 |
| austria_ring_1,Ni=16,delta=0.10 | 0.0922 | 0.1029 | +0.0107 |
| austria_ring_2,Ni=4,delta=0.05 | 0.1897 | 0.2032 | +0.0135 |
| austria_ring_2,Ni=4,delta=0.10 | 0.1810 | 0.2019 | +0.0209 |
| austria_ring_2,Ni=16,delta=0.05 | 0.1260 | 0.2119 | +0.0859 |
| austria_ring_2,Ni=16,delta=0.10 | 0.1367 | 0.2040 | +0.0673 |

## Section 5.2.3 mixed_circle

| metric | value |
|---|---:|
| Ni=10,delta=0.05,rel_l2 | 0.1308 |
| Ni=10,delta=0.05,acc | 0.9818 |
| Ni=10,delta=0.10,rel_l2 | 0.1425 |
| Ni=10,delta=0.10,acc | 0.9816 |

## Notes

- No public author repo was found; protocol/parameters follow arXiv:2403.02584v2.
- DSM-DL inputs are computed via Eq. (3.11)/(3.12) from a Helmholtz Lippmann-Schwinger solve.
- Training noise = 1%; test noise per Eq. (5.1) at each evaluated delta.
- MNIST uses official torchvision train/test split when reachable; otherwise a by-description fallback is used and disclosed in the JSON metadata.
- Chinese-like and Austria-like profiles are constructed from the paper's textual description; the original test images are not public.