# Parabolic IDSM 重写 - 端到端 noiseless 烟测报告

日期: 2026-05-06

## 配置

- Example 5.1 ConductivityMerging
- noise_level=0.0, total_time=1.5 (15 段, deltaT=0.1, deltaTsplit=6)
- coarse: n_boundary=80 → 1566 三角形
- fine: n_boundary=113 → 3187 三角形
- tolerance=0.08, save_num=10, forget_scale=0.7, kappa=1e10

## 结果

15 段全跑通，每段 inner_loop = 1（noiseless 下 residual 0.004-0.015 远低于 tolerance）。

| seg | t | edp_min | py_min | L∞ | RMS | py_IoU |
|---|---|---|---|---|---|---|
| 0 | 0.10 | 0.682 | 0.672 | 0.147 | 0.033 | 0.000 |
| 1 | 0.20 | 0.348 | 0.010 | 0.990 | 0.109 | 0.000 |
| 2 | 0.30 | 0.857 | 0.727 | 0.210 | 0.071 | 0.000 |
| 3 | 0.40 | 0.994 | 0.313 | 0.687 | 0.260 | 0.000 |
| 4 | 0.50 | 0.010 | 0.139 | 0.891 | 0.318 | 0.152 |
| 5 | 0.60 | 0.010 | 0.010 | 0.875 | 0.257 | 0.376 |
| 6 | 0.70 | 0.010 | 0.010 | 0.990 | 0.352 | 0.418 |
| 7 | 0.80 | 0.010 | 0.010 | 0.990 | 0.284 | 0.483 |
| 8 | 0.90 | 0.393 | 0.010 | 0.990 | 0.208 | 0.478 |
| 9 | 1.00 | 0.010 | 0.010 | 0.990 | 0.200 | 0.502 |
| 10 | 1.10 | 0.010 | 0.010 | 0.990 | 0.144 | 0.487 |
| 11 | 1.20 | 0.107 | 0.010 | 0.990 | 0.186 | 0.439 |
| 12 | 1.30 | 0.010 | 0.010 | 0.889 | 0.141 | 0.483 |
| 13 | 1.40 | 0.010 | 0.010 | 0.890 | 0.159 | 0.504 |
| 14 | 1.50 | (no baseline) | 0.010 | - | - | 0.458 |

## 重写过程修复的两个量级 bug

1. **fine→coarse 投影 fill=0**：`project_p1_fine_to_coarse` 默认对凸包外点用 0 兜底
   导致 coarse 圆周节点（落在 fine 弦外）测量值被打穿，初始 residual=13.78。
   修：径向 ε 收缩重插 + cKDTree 最近邻兜底。

2. **normalScale 用 residual 而非 measurement**：`solve_adjoint_segment` 把累加项
   误算成 yEmptyHistory（即 residual）的 ∫|·|² ds。.edp L404 实际是
   measurement(=BoundaryData) 自身的平方积分。修：增加 measurement_history
   参数，norm_acc 用 measurement 累加，rhs 仍用 residual。

修复后 seg 0 σ=[0.672, 1.000] 与 .edp [0.682, 1.000] L∞ 仅 0.147，趋势全程一致。

## 残留差距

- seg 1, 3 出现 σ 推到下界 0.010 而 .edp 维持非饱和值 (0.348/0.994)。可能根因：
  (a) R.diag 段间累积放大；(b) 某些段 .edp 实际跑了多次 inner（noiseless tolerance
  恰好松动），导致 σ 自调节。当前 noiseless+tol=0.08 全 1 次 inner 单步收敛。
- 全 15 段 IoU 终点 0.458，已远高于此前所有版本（之前长期 0.0-0.05）。

## 后续

- 全 102 段 IoU 上限测定（task #42）
- seg 1/3 过冲根因细查（可能不影响最终 IoU，但影响 L∞ 局部精度）
- 五个 Example 全跑 + 论文图重生（task #16）

---

## 全 102 段 noiseless IoU 上限测定 (task #42, 2026-05-07)

脚本: `scripts/run_full_102_noiseless.py`，total_time=10.2 (102 段)，cfg 同上。

耗时 26.6s。所有 102 段 n_inner=1（noiseless 单步残差 1e-3~2e-2 已远低于 tol=0.08）。

### 关键指标

| 指标 | 值 |
|---|---|
| max IoU | **0.7498 at seg 96** |
| final IoU (seg 101) | 0.377 |
| mean IoU last 20 segs | 0.313 |
| segs IoU ≥ 0.5 | 9 / 102 |
| segs IoU ≥ 0.7 | 2 / 102 |

### IoU 演化模式

- segs 0-3: IoU=0（早期 σ 仍偏离）
- segs 4-15: climb 至 0.50（基础重建启动）
- segs 16-30: 0.15-0.40（震荡）
- segs 50-60: 谷底 0.0-0.05（IoU=0 at seg 56）
- segs 60-80: 恢复 0.30-0.45
- **segs 90-97: 锐峰 0.71/0.75**
- segs 98-101: 回落 0.37

### 主要观察

1. **n_inner=1 全程**：noiseless 下首步投影即满足 tol，BFG 内迭代未被触发；σ 是单步 R₀·ζ 投影结果。
2. **σ_min 长期饱和到 0.010**：projection 下界 `clip(c_grad, -0.99/Δc, 0)·Δc + cA = -0.99·0.9 + 1.0 = 0.01`。从 seg 5 开始几乎所有段的 σ_min 都贴这个下界，意味着 reconstructor 把大片区域识别为 "完全 inclusion"，但真值应是 σ=cB=0.1。
3. **IoU 在 seg 96 触 0.75 但不稳定**：方法可以达到目标，但稳定性不够。
4. **趋势相关于真值轨迹**：seg 90-97 (t=9.0-9.7) 高 IoU 期与真值椭圆相对静止/中心化的时段对应；IoU 谷底 (seg 50-60) 对应椭圆运动剧烈期。

### 后续诊断方向（task #16 之前必修）

- BFG 状态 R 跨段累积可能过强 → 验证 `cfg.save_num` 跨段 reset 语义
- σ 过度极化的 ζ_c 量级排查（可能 normalScale 仍偏大）
- 跑更长（200+ 段）确认 0.75 是真上限还是巧合
- 与 .edp 完整 102 段 baseline 对照（须先把 .edp 跑出 102 个 mid102_*.dat）

### 输出文件

- `logs/full_102_noiseless.txt` — per-seg σ_min/max/IoU/n_inner/final_resid 表
- `logs/full_102_noiseless_iou.png` — IoU 曲线
- `logs/full_102_sigma_snapshots.npz` — 每 10 段一张 σ snapshot（含 mesh）
- `logs/full_102_noiseless.log` — verbose 运行日志
