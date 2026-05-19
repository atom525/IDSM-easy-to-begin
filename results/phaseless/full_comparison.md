# Full Phaseless Strict Reproduction Comparison

- paper: `arXiv:2403.02584v2`
- device: `cuda`
- seed: `0`
- protocol: strict (Helmholtz forward + Eq. 3.11 DSM input, no proxy)

## Table 1 (Polygon accuracy)

| case | paper | ours | diff |
|---|---:|---:|---:|
| Ni=1,delta=0.02 | 0.9949 | 0.9873 | -0.0076 |
| Ni=1,delta=0.10 | 0.9772 | 0.9793 | +0.0021 |
| Ni=4,delta=0.02 | 0.9977 | 0.9860 | -0.0117 |
| Ni=4,delta=0.10 | 0.9916 | 0.9845 | -0.0071 |

## Table 2 (Relative L2)

| case | paper | ours | diff |
|---|---:|---:|---:|
| mnist,Ni=4,delta=0.05 | 0.0827 | 0.0970 | +0.0143 |
| mnist,Ni=4,delta=0.10 | 0.1043 | 0.1051 | +0.0008 |
| mnist,Ni=16,delta=0.05 | 0.0617 | 0.0804 | +0.0187 |
| mnist,Ni=16,delta=0.10 | 0.0755 | 0.0872 | +0.0117 |
| chinese_like,Ni=4,delta=0.05 | 0.1096 | 0.1361 | +0.0265 |
| chinese_like,Ni=4,delta=0.10 | 0.1252 | 0.1438 | +0.0186 |
| chinese_like,Ni=16,delta=0.05 | 0.0721 | 0.1276 | +0.0555 |
| chinese_like,Ni=16,delta=0.10 | 0.0854 | 0.1387 | +0.0533 |
| austria_ring_1,Ni=4,delta=0.05 | 0.1163 | 0.0949 | -0.0214 |
| austria_ring_1,Ni=4,delta=0.10 | 0.1258 | 0.1053 | -0.0205 |
| austria_ring_1,Ni=16,delta=0.05 | 0.0851 | 0.0967 | +0.0116 |
| austria_ring_1,Ni=16,delta=0.10 | 0.0922 | 0.1023 | +0.0101 |
| austria_ring_2,Ni=4,delta=0.05 | 0.1897 | 0.2135 | +0.0238 |
| austria_ring_2,Ni=4,delta=0.10 | 0.1810 | 0.2030 | +0.0220 |
| austria_ring_2,Ni=16,delta=0.05 | 0.1260 | 0.2094 | +0.0834 |
| austria_ring_2,Ni=16,delta=0.10 | 0.1367 | 0.2071 | +0.0704 |

## Section 5.2.3 mixed_circle

| metric | value |
|---|---:|
| Ni=10,delta=0.05,rel_l2 | 0.1655 |
| Ni=10,delta=0.05,acc | 0.9765 |
| Ni=10,delta=0.10,rel_l2 | 0.1888 |
| Ni=10,delta=0.10,acc | 0.9742 |

## Notes

- No public author repo was found; protocol/parameters follow arXiv:2403.02584v2.
- DSM-DL inputs are computed via Eq. (3.11)/(3.12) from a Helmholtz Lippmann-Schwinger solve.
- Training noise = 1%; test noise per Eq. (5.1) at each evaluated delta.
- MNIST uses official torchvision train/test split when reachable; otherwise a by-description fallback is used and disclosed in the JSON metadata.
- Chinese-like and Austria-like profiles are constructed from the paper's textual description; the original test images are not public.