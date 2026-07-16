# Full Phaseless Strict Reproduction Comparison

- paper: `arXiv:2403.02584v2`
- device: `cuda`
- seed: `0`
- protocol: strict (Helmholtz forward + Eq. 3.11 DSM input, no proxy)

## Table 1 (Polygon accuracy)

| case | paper | ours | diff |
|---|---:|---:|---:|
| Ni=1,delta=0.02 | 0.9949 | 0.9933 | -0.0016 |
| Ni=1,delta=0.10 | 0.9772 | 0.9823 | +0.0051 |
| Ni=4,delta=0.02 | 0.9977 | 0.9966 | -0.0011 |
| Ni=4,delta=0.10 | 0.9916 | 0.9910 | -0.0006 |

## Table 2 (Relative L2)

| case | paper | ours | diff |
|---|---:|---:|---:|
| mnist,Ni=4,delta=0.05 | 0.0827 | 0.0827 | +0.0000 |
| mnist,Ni=4,delta=0.10 | 0.1043 | 0.1174 | +0.0131 |
| mnist,Ni=16,delta=0.05 | 0.0617 | 0.0603 | -0.0014 |
| mnist,Ni=16,delta=0.10 | 0.0755 | 0.0873 | +0.0118 |
| chinese_like,Ni=4,delta=0.05 | 0.1096 | 0.1192 | +0.0096 |
| chinese_like,Ni=4,delta=0.10 | 0.1252 | 0.1387 | +0.0135 |
| chinese_like,Ni=16,delta=0.05 | 0.0721 | 0.0799 | +0.0078 |
| chinese_like,Ni=16,delta=0.10 | 0.0854 | 0.1113 | +0.0259 |
| austria_ring_1,Ni=4,delta=0.05 | 0.1163 | 0.1064 | -0.0099 |
| austria_ring_1,Ni=4,delta=0.10 | 0.1258 | 0.1622 | +0.0364 |
| austria_ring_1,Ni=16,delta=0.05 | 0.0851 | 0.0656 | -0.0195 |
| austria_ring_1,Ni=16,delta=0.10 | 0.0922 | 0.1096 | +0.0174 |
| austria_ring_2,Ni=4,delta=0.05 | 0.1897 | 0.1217 | -0.0680 |
| austria_ring_2,Ni=4,delta=0.10 | 0.1810 | 0.1398 | -0.0412 |
| austria_ring_2,Ni=16,delta=0.05 | 0.1260 | 0.1000 | -0.0260 |
| austria_ring_2,Ni=16,delta=0.10 | 0.1367 | 0.1278 | -0.0089 |

## Section 5.2.3 mixed_circle

| metric | value |
|---|---:|
| Ni=10,delta=0.05,rel_l2 | 0.1029 |
| Ni=10,delta=0.05,acc | 0.9824 |
| Ni=10,delta=0.10,rel_l2 | 0.1135 |
| Ni=10,delta=0.10,acc | 0.9844 |

## Notes

- No public reference repo was found; protocol/parameters follow arXiv:2403.02584v2.
- DSM-DL inputs are computed via Eq. (3.11)/(3.12) from a Helmholtz Lippmann-Schwinger solve.
- Training noise = 1%; test noise per Eq. (5.1) at each evaluated delta.
- MNIST uses official torchvision train/test split when reachable; otherwise a by-description fallback is used and disclosed in the JSON metadata.
- Chinese-character profiles are extracted from the visible truth row of paper Fig. 8; Austria-like profiles are constructed from the paper's textual description.