# GeoSAR-Forge 当前正式流程技术文档

> 当前正式主链已经更新为  
> `ISCE2 → Dolphin → 主链 QC / DePSI-like QC → MintPy(pass1→feedback→pass2) → 导出 → support_graph_v1 异常形变区识别 → MintPy 后未来预测 v2`

相关补充文档：
- [DePSI_like_QC_创新说明.md](/data/InSAR/docs/DePSI_like_QC_创新说明.md)
- [MintPy后高可信点预测说明.md](/data/InSAR/docs/MintPy后高可信点预测说明.md)
- [用户手册.md](/data/InSAR/docs/用户手册.md)

## 1. 总体结构

当前系统分成三段：

1. 主 InSAR 反演链  
   `AOI / 场景筛选 → 大气方案 → DEM → SLC 下载 → ISCE2 → Dolphin → Mainchain QC / DePSI-like QC → MintPy 两阶段反演`
2. 区域识别链  
   在最终接受的 MintPy 结果和高可信点骨架上运行 `support_graph_v1`，自动输出异常形变区 polygon、区域统计和区域级累计形变曲线。
3. MintPy 后预测链  
   在最终接受的 MintPy 时序上构建分解感知时序样本，训练 `generic / hazard` 双模式预测器，并用 conformal 校准输出预测区间。

要点：
- 形变区识别已经不再是 `weak_ml_v1` 像素分类默认链，而是高可信点图上的区域分割。
- 预测默认不再被 zone 限制，当前默认 `forecast_point_scope = all_high_confidence`。
- `generic` 不再是旧版单一 `GRU` 主线，而是 `TemporalFusionForecaster` 与 `DecompTCNGRUQuantileForecaster` 同训后按验证集 MAE 择优。

## 2. Notebook 顺序

当前 [pipeline.ipynb](/data/InSAR/pipeline.ipynb) 的顺序是：

| 单元 | 作用 |
|------|------|
| Cell 1 | 项目初始化与全局配置 |
| Cell 2 | AOI / 场景筛选 |
| Cell 3 | 大气方案选择 + DEM 准备 |
| Cell 4 | SLC 下载 + ISCE2 |
| Cell 5 | Dolphin |
| Cell 5A | 主链 QC / DePSI-like QC |
| Cell 6 | MintPy 两阶段反演 |
| Cell 7 | 可视化与导出 |
| Cell 8 | `support_graph_v1` 异常形变区识别 |
| Cell 9 | 未来预测训练 |
| Cell 10 | 预测推理 / 评估 / 绘图 |
| Cell 11 | 可选清理 |

推荐执行方式：
- 只要主 InSAR 结果：`Cell 1 → 7`
- 要区域识别：继续跑 `Cell 8`
- 要未来预测：继续跑 `Cell 9 → 10`

## 3. 主 InSAR 反演链

### 3.1 AOI、场景筛选与时间均匀采样

AOI 解析和 ASF 搜索之后，系统按时间均匀性和垂直基线分布筛选场景。理想时间间隔写成：

$$
\Delta t_{ideal} = \frac{t_{end} - t_{start}}{N_{target} - 1}
$$

对每个理想时隙，按下式选场景：

$$
score(j)=|t_j-t_i| + w\cdot |B_{\perp,j}-\tilde B_\perp|
$$

其中：
- $B_{\perp,j}$ 是垂直基线
- $\tilde B_\perp$ 是中位垂直基线
- $w$ 用来平衡时间均匀性和基线分布

### 3.2 大气方案前移

大气方案在 SLC 下载前确定，支持：

- `GACOS`
- `ERA5`
- `MERRA-2`
- `height_correlation`
- `spatial_filter`
- `no`

当前阶段只固化方案与配置；真正下载或预处理在 MintPy 前的主链 QC 阶段统一执行。

### 3.3 ISCE2 与 Dolphin

ISCE2 负责几何配准、轨道、重采样与 stack 准备。  
Dolphin 负责 PS/DS phase linking、网络型干涉图构建和 `SNAPHU` 解缠，输出：

- `*.unw.tif`
- `*.int.cor.tif`
- `*.unw.conncomp.tif`
- `temporal_coherence_average_*.tif`

### 3.4 主链 QC 与 DePSI-like QC

#### Pair 级 QC

Pair 风险分数为：

$$
risk = 0.25(1-\mathrm{norm\_coh})
+ 0.20(1-\mathrm{valid\_conncomp\_frac})
+ 0.15\,\mathrm{norm}(n\_conncomp)
+ 0.20\,\mathrm{norm}(p95\_abs\_los)
+ 0.20\,\mathrm{norm}(frac_{>50mm})
$$

pair 会被分成 `keep / downweight / drop`，并生成：

- `pair_qc.csv`
- `pair_qc_summary.json`
- `ifgramStack_qc.h5`

#### DePSI-like 高可信骨架

当前 `ps_score` 定义为：

$$
ps\_score =
0.30\,tcoh +
0.20\,valid\_pair\_ratio +
0.20\,mainCC\_ratio +
0.15(1-model\_rms\_norm) +
0.15(1-jump\_risk)
$$

默认阈值：

$$
\tau_{strict}=0.75,\qquad \tau_{relaxed}=0.60
$$

并输出：

- `ps_score.tif`
- `mask_ps_strict.tif`
- `mask_ps_relaxed.tif`
- `ps_model_metrics.h5`
- `ps_score_summary.json`
- `ref_primary_network.csv/.tif`
- `ref_candidates.csv`

#### 参考候选

参考候选核心评分：

$$
ref\_score =
0.35\,tcoh +
0.25\,mainCC\_ratio +
0.20(1-model\_rms\_norm) +
0.10\,valid\_pair\_ratio +
0.10\,neighbor\_stability
$$

并结合：
- `dist_to_edge`
- `strict_neighbor_count`
- `local_velocity_gradient`
- `gacos_safe_flag`

### 3.5 MintPy 两阶段时序反演

当前 MintPy 是：

1. `pass1`
2. `feedback`
3. `pass2`

异常日期会记录：
- `median_abs_residual_mm`
- `frac_gt_threshold`
- `spatial_concentration_index`
- `anomaly_scope ∈ {global, mixed, local}`

`pass2` 接受条件：

$$
strict\_PS\_adjacent\_jump\_ratio(pass2)\le strict\_PS\_adjacent\_jump\_ratio(pass1)
$$

$$
strict\_PS\_median\_model\_rms(pass2)\le 1.05\times strict\_PS\_median\_model\_rms(pass1)
$$

$$
strict\_PS\_retained\_count(pass2)\ge 0.85\times strict\_PS\_retained\_count(pass1)
$$

不满足时正式结果自动回退到 `pass1`。

## 4. 异常形变区识别：`support_graph_v1`

### 4.1 入口与默认值

区域识别入口是：

`detect_deformation_zones(..., detector_mode="support_graph_v1", zone_semantics="anomalous_deformation", forecast_point_scope="all_high_confidence")`

当前默认不再使用 `weak_ml_v1` 作为正式检测主干；`weak_ml_v1` 仅保留为显式 legacy 模式。

### 4.2 支持点集合

区域检测不再在全栅格上直接分类，而是先在 `strict ∪ relaxed` 支持点上建图。  
对每个支持点，从最终接受的 `rel0` 时序提取：

- `net_disp_mm`
- `peak_abs_disp_mm`
- `trend_mm_yr`
- `path_length_mm`
- `net_to_path_ratio`
- `sign_consistency`
- `pc1 / pc2 / pc3`
- `local_contrast_net`
- `local_contrast_peak`
- `neighbor_trace_corr`
- `support_density`
- `qc_support`

其中：

$$
path\_length = \sum_t |\Delta x_t|\cdot \mathbf{1}(|\Delta x_t|>0.25)
$$

$$
sign\_consistency = \frac{\max(N_{+},N_{-})}{N_{+}+N_{-}}
$$

$$
net\_to\_path\_ratio = \frac{|x_T-x_0|}{path\_length}
$$

$$
temporal\_coherence = 0.5\cdot net\_to\_path\_ratio + 0.5\cdot sign\_consistency
$$

$$
qc\_support =
0.35\,ps\_score +
0.25\,tcoh +
0.20\,valid\_pair\_ratio +
0.20\,mainCC\_ratio
$$

### 4.3 Saliency 与图结构

记鲁棒 z-score 为 $\rho(\cdot)$，Sigmoid 为 $\sigma(\cdot)$。  
活动性和对比度的 logit 为：

$$
activity\_logit =
0.45\,\rho(peak\_abs\_disp) +
0.35\,\rho(|net\_disp|) +
0.20\,\rho(|trend|)
$$

$$
contrast\_logit =
0.60\,\rho(local\_contrast\_peak) +
0.40\,\rho(local\_contrast\_net)
$$

$$
activity = \sigma(activity\_logit)
$$

$$
saliency = \sigma\Big(
0.42\,activity\_logit +
0.24\,contrast\_logit +
1.10(temporal\_coherence-0.45) +
0.55(neighbor\_trace\_corr-0.35) +
0.35(support\_density-0.45) +
0.25(qc\_support-0.55)
\Big)
$$

边保留条件是：

$$
sign_i = sign_j,\qquad
d_{ij}\le \max(4.5\tilde d_{nn}, 0.10\ \text{km}),\qquad
corr(z_i,z_j)\ge 0.58
$$

其中 $\tilde d_{nn}$ 是最近邻中位距离。

### 4.4 Core / Active 生长

代码里的默认阈值是：

- `SUPPORT_GRAPH_K = 12`
- `active saliency >= max(0.50, P80(saliency))`
- `core saliency >= max(0.62, P92(saliency))`
- `active coherence >= 0.48`
- `core coherence >= 0.55`
- `qc_support >= 0.50`

即：

$$
active =
\mathbf{1}(saliency\ge \max(0.50,P80))
\cdot \mathbf{1}(coherence\ge 0.48)
\cdot \mathbf{1}(qc\_support\ge 0.50)
$$

$$
core =
active \cdot
\mathbf{1}(saliency\ge \max(0.62,P92))
\cdot \mathbf{1}(activity\ge P75(activity))
\cdot \mathbf{1}(coherence\ge 0.55)
$$

候选区域必须在 `active` 子图内与 `core` 连通，否则直接丢弃。

### 4.5 区域评分与保留

候选区域评分为：

$$
region\_score =
\sqrt{area\_fraction}\cdot
internal\_trace\_corr\cdot
temporal\_coherence\cdot
\max(boundary\_contrast, 0.05)\cdot
\log(1+activity\_level)
$$

其中：

$$
boundary\_contrast =
\frac{inside\_level-outside\_level}{\max(inside\_level, 1.0)}
$$

默认保留门槛：

- `area_fraction >= max(8e-4, 0.25 * largest_candidate_area_fraction)`
- `support_point_count >= max(256, 0.003 * n_support_points)`
- `internal_trace_corr >= 0.60`
- `temporal_coherence >= 0.55`
- `compactness >= 0.20`
- `region_score >= 0.35 * best_region_score`

此外还保留 `salient_exception` 例外通道，并允许相邻同类候选在高相关下自动合并。

### 4.6 栅格 outward growth

保留下来的支持点区域不会直接停止在骨架上，而是向真实 MintPy 栅格做 outward growth。  
以区域中心线与全图像元时序相关系数作为传播约束：

$$
corr_i = \frac{\langle \tilde x_i,\tilde x_{zone}\rangle}{\|\tilde x_i\|\cdot \|\tilde x_{zone}\|}
$$

并对不同 `zone_type` 使用符号一致的 `net / peak / velocity / coherence` 门槛扩张，最终得到：

- `deformation_zone_probability.tif`
- `deformation_zone_mask.tif`
- `deformation_zone_id.tif`
- `deformation_zones.geojson/.shp/.kmz`
- `deformation_zone_summary.json`
- `deformation_zone_timeseries.csv`
- `velocity_map_zones.png/.pdf`

当前区域级曲线使用 `20% trimmed mean` 作为中心线，`p25-p75` 作为带状区间。

## 5. MintPy 后未来预测 v2

### 5.1 数据层与分解

预测默认读取最终接受的 MintPy 时序 `rel0`：

$$
x_{rel0}(t)=x(t)-x(t_0)
$$

并在每个点上拟合固定的 `robust_harmonic_v1`：

$$
x_{rel0}(t)=\beta_0+\beta_1 t + a_1\sin\frac{2\pi t}{365.25}+a_2\cos\frac{2\pi t}{365.25}+r(t)
$$

从而得到：
- `trend_component`
- `seasonal_component`
- `residual_component`

序列通道固定为 16 个：

- `raw`: `rel0 / delta / day_gap / sin_doy / cos_doy`
- `decomposition`: `trend_component / seasonal_component / residual_component / delta2 / rolling_velocity_3step / rolling_residual_std`
- `neighborhood`: `neighbor_mean_rel0 / neighbor_mean_delta / neighbor_delta_std`
- `event`: `local_event_persistence / abnormal_date_flag`

静态特征固定为：

- `ps_score`
- `tcoh`
- `valid_pair_ratio`
- `mainCC_ratio`
- `jump_risk`
- `anomaly_exposure`
- `velocity_mm_yr`
- `height_m`

### 5.2 预测点集与 forecast scope

点集自动策略仍是三段式：

1. `strict_only`
2. `strict_plus_relaxed`
3. `fallback_baseline_only`

但 `forecast_point_scope` 当前默认是：

$$
forecast\_point\_scope = all\_high\_confidence
$$

也就是说：
- 默认预测全部 DePSI-like 高可信点
- zone 结果默认只作为区域识别、区域统计和点级附加字段
- 只有显式设置 `zone_high_confidence_only` 才会改成 `strict/relaxed ∩ zone_mask`

### 5.3 预测图结构与 hazard fallback

预测邻域图是单独的 kNN 图，默认：

- `graph_k = 8`
- 边特征：`distance_m / elevation_diff_m / velocity_diff_mm_yr / ps_score_diff`

`hazard` 结构性 fallback 规则来自数据层：

- `effective_nodes < max(64, 4*k)`
- `mean_degree < 3`
- `largest_component_ratio < 0.60`
- `hazard_train_windows_est < 0.5 * generic_train_windows_est`
- `neighbor_nan_ratio > 0.20`

### 5.4 模型：`generic / hazard`

#### Generic：两模型竞争，验证集择优

`generic` 不是单一模型，而是同时训练：

1. `TemporalFusionForecaster`
2. `DecompTCNGRUQuantileForecaster`

然后按验证集 `p50` 的 MAE 选出 canonical `generic_forecaster.pt`。

#### `TemporalFusionForecaster`

输入嵌入写成：

$$
h_t^{(0)} = \mathrm{Proj}(s_t) + \mathrm{PE}\!\left(\sum_{\tau\le t}\Delta day_\tau\right)
$$

再经过多层多头自注意力：

$$
h_t^{(\ell+1)} = \mathrm{AttnBlock}(h_t^{(\ell)})
$$

时间池化不是简单取末端，而是门控平均：

$$
g_t = \sigma(W_g h_t),\qquad
h_{temp} = \frac{1}{T}\sum_t g_t \odot h_t
$$

若启用邻域上下文，则对邻居汇聚：

$$
\alpha_j = \mathrm{softmax}(score(h_{temp}, n_j, e_j))
$$

$$
h'_{temp}=h_{temp} + gate(h_{temp}, \sum_j \alpha_j n_j)
$$

最后与静态特征融合输出分位数。

#### `DecompTCNGRUQuantileForecaster`

长分支取 `raw + decomposition`：

$$
h_{long} = \mathrm{SelfAttn}\big(\mathrm{TCN}(\mathrm{Proj}(x_{raw+decomp}))\big)_{t=T}
$$

短分支取 `residual + event + neighborhood`：

$$
h_{short} = \mathrm{GRU}(\mathrm{Proj}(x_{short}))_{t=T}
$$

静态特征走独立 MLP，最后用门控融合：

$$
g = \sigma\!\left(W[h_{long};h_{short};h_{static}]\right)
$$

$$
h_{fused}=g\odot h_{long} + (1-g)\odot h_{short}
$$

#### Hazard：`GraphTCNAttnQuantileForecaster`

`hazard` 仅在结构门槛满足时训练。  
它对每个时间步做 edge-aware 邻域聚合，然后走 TCN 和时间注意力池化：

$$
\alpha_{t,j}=\mathrm{softmax}(score(c_t, n_{t,j}, s_j, e_j))
$$

$$
\tilde c_t = c_t + \phi\!\left(c_t,\sum_j \alpha_{t,j} n_{t,j}\right)
$$

$$
h = \mathrm{AttnPool}(\mathrm{TCN}(\tilde c_{1:T}))
$$

### 5.5 训练目标、loss 与单调约束

内部训练目标仍是 future offset：

$$
\mathbf y_{offset}=[x(t+1)-x(t),\dots,x(t+H)-x(t)]
$$

模型输出 `p10 / p50 / p90`，loss 是 pinball quantile loss 加单调惩罚：

$$
\mathcal L_{quantile}
=
\frac{1}{NHTQ}\sum \max(q\cdot e, (q-1)\cdot e)
+ \lambda\Big(\mathrm{ReLU}(q_{10}-q_{50})+\mathrm{ReLU}(q_{50}-q_{90})\Big)
$$

其中：
- $e=y-\hat y_q$
- $Q=\{0.1,0.5,0.9\}$
- 当前代码里 `λ = 0.1`

推理后还会再做一次：

$$
[q_{10},q_{50},q_{90}] \leftarrow sort([q_{10},q_{50},q_{90}])
$$

### 5.6 Baseline

当前固定 baseline 为四条：

- `persistence`
- `linear_trend`
- `seasonal_naive`
- `harmonic_trend`

### 5.7 CQR conformal 校准

当前正式不确定性模式是：

$$
uncertainty\_mode = cqr\_conformal\_v1
$$

在验证集上，对每个 horizon 单独计算 nonconformity：

$$
s_h^{(n)} = \max(q_{10,raw}^{(n)}-y^{(n)},\ y^{(n)}-q_{90,raw}^{(n)},\ 0)
$$

再取 split-conformal 分位数修正量 $\delta_h$，并得到：

$$
q_{10,cal}=q_{10,raw}-\delta_h,\qquad
q_{50,cal}=q_{50,raw},\qquad
q_{90,cal}=q_{90,raw}+\delta_h
$$

当前目标覆盖率：

$$
target\_coverage = 0.80
$$

### 5.8 `c_pred` 不是模型直接输出

当前 `c_pred` 是 calibrated interval 宽度驱动的风险感知代理分数：

$$
width\_norm = \mathrm{norm}(interval\_width_{cal})
$$

$$
c_{pred} = reliability \cdot
\Big[
0.60(1-width\_norm)
+ 0.20\,ps\_score
+ 0.10(1-jump\_risk)
+ 0.10(1-anomaly\_exposure)
\Big]
$$

这里的 `reliability` 就是验证集 `calibration_reliability_factor`。

### 5.9 输出

训练输出：

- `forecast_train_summary.json`
- `forecast_data_summary.json`
- `forecast_calibration.json`
- `forecast_explainability.json`
- `forecast_normalizer.joblib`

推理输出：

- `forecast_predictions.h5`
- `forecast_summary.csv`
- `forecast_point_inventory.csv`

评估输出：

- `forecast_evaluation.json`
- `forecast_baseline_comparison.csv`

图件输出：

- `forecast_confidence_map.png`
- `forecast_horizon_panel.png`
- `forecast_calibration_curve.png`
- `forecast_feature_group_importance.png`
- `forecast_neighbor_influence_map.png`（仅 hazard 有效时）

## 6. 当前代码里的几个默认事实

这几件事很容易被旧文档误导，单独列出：

- 默认区域检测主干是 `support_graph_v1`，不是 `weak_ml_v1`
- 默认预测范围是 `all_high_confidence`，不是 `zone_high_confidence_only`
- `generic` 主模式不是“固定 GRU”，而是 `TemporalFusion` 与 `DecompTCNGRU` 竞争后择优
- `hazard` 不是默认主模式，且可能因为图结构或验证表现回退到 `generic`
- canonical `pred_offset_* / pred_rel0_*` 现在写的是校准后区间；原始分位数保存在 `*_raw`
