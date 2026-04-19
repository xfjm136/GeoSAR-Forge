# GeoSAR-Forge MintPy 后未来预测说明

这份文档只描述当前 **真实代码** 里的 MintPy 后链，不再沿用旧版“单点短窗 GRU + 启发式置信度”的叙述。

当前正式后链是：

$$
\text{MintPy accepted result}
\rightarrow
\text{forecast data layer}
\rightarrow
\text{generic / hazard models}
\rightarrow
\text{CQR calibration}
\rightarrow
\text{evaluation / figures / inventory}
$$

其中：
- `generic` 是默认主模式
- `hazard` 是图结构增强模式
- 默认预测范围是 `all_high_confidence`
- 形变区识别结果默认只作为附加信息，不强制限制预测点

## 1. 预测输入：分解感知时序 + 邻域 + QC 静态特征

### 1.1 基础时序语义

内部统一使用：

$$
x_{rel0}(t)=x(t)-x(t_0)
$$

再构造：

$$
\Delta x(t)=x_{rel0}(t)-x_{rel0}(t-1)
$$

并保留实际 SAR revisit 间隔：

$$
\Delta day(t)=day\_offset(t)-day\_offset(t-1)
$$

### 1.2 `robust_harmonic_v1`

每个点都拟合同一套鲁棒谐波模型：

$$
x_{rel0}(t)=\beta_0+\beta_1 t + a_1\sin\frac{2\pi t}{365.25}+a_2\cos\frac{2\pi t}{365.25}+r(t)
$$

得到：

- `trend_component`
- `seasonal_component`
- `residual_component`

此外还会继续派生：

- `delta2`
- `rolling_velocity_3step`
- `rolling_residual_std`
- `local_event_persistence`

### 1.3 固定通道顺序

当前序列通道固定是 16 维：

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

## 2. 预测点集不是简单“全图都做”

当前点集选择逻辑仍然是三段式：

1. `strict_only`
2. `strict_plus_relaxed`
3. `fallback_baseline_only`

但和旧文档不同的是，当前默认：

$$
forecast\_point\_scope = all\_high\_confidence
$$

只有显式设为 `zone_high_confidence_only` 时，才会把训练/推理限制为：

$$
(\text{strict} \cup \text{relaxed}) \cap zone\_mask
$$

## 3. 图结构与 hazard 可用性

预测链也会构建一个单独的 kNN 图，默认：

- `graph_k = 8`
- 边特征：`distance_m / elevation_diff_m / velocity_diff_mm_yr / ps_score_diff`

`hazard` 只有在以下条件都合理时才可训练：

- `effective_nodes >= max(64, 4*k)`
- `mean_degree >= 3`
- `largest_component_ratio >= 0.60`
- `hazard_train_windows_est >= 0.5 * generic_train_windows_est`
- `neighbor_nan_ratio <= 0.20`

否则自动 fallback 到 `generic`。

## 4. 当前真实模型

### 4.1 `generic` 不是单一 GRU

代码里会同时训练两套 generic 候选：

1. `TemporalFusionForecaster`
2. `DecompTCNGRUQuantileForecaster`

然后比较验证集 `p50` 的 MAE，择优写成 canonical：

`generic_forecaster.pt`

### 4.2 `TemporalFusionForecaster`

其核心是：

$$
h_t^{(0)}=\mathrm{Proj}(s_t)+\mathrm{PE}\!\left(\sum_{\tau\le t}\Delta day_\tau\right)
$$

$$
h_t^{(\ell+1)}=\mathrm{AttnBlock}(h_t^{(\ell)})
$$

$$
g_t=\sigma(W_g h_t),\qquad
h_{temp}=\frac{1}{T}\sum_t g_t\odot h_t
$$

如果邻域上下文可用，还会额外做一次轻量邻居注意力汇聚。

### 4.3 `DecompTCNGRUQuantileForecaster`

它把时序拆成长短两条支路：

长支路：

$$
h_{long}=\mathrm{SelfAttn}\big(\mathrm{TCN}(\mathrm{Proj}(x_{raw+decomp}))\big)_{t=T}
$$

短支路：

$$
h_{short}=\mathrm{GRU}(\mathrm{Proj}(x_{residual+event+neighbor}))_{t=T}
$$

再与静态特征门控融合：

$$
g=\sigma(W[h_{long};h_{short};h_{static}])
$$

$$
h_{fused}=g\odot h_{long}+(1-g)\odot h_{short}
$$

### 4.4 `hazard = GraphTCNAttnQuantileForecaster`

`hazard` 不是全图 Transformer，而是局部图聚合 + TCN：

$$
\alpha_{t,j}=\mathrm{softmax}(score(c_t,n_{t,j},s_j,e_j))
$$

$$
\tilde c_t=c_t+\phi\!\left(c_t,\sum_j \alpha_{t,j}n_{t,j}\right)
$$

$$
h=\mathrm{AttnPool}(\mathrm{TCN}(\tilde c_{1:T}))
$$

## 5. 训练目标与 loss

内部训练目标仍是 future offset：

$$
\mathbf y_{offset}=[x(t+1)-x(t),\dots,x(t+H)-x(t)]
$$

模型直接输出：

$$
[\hat q_{0.1},\hat q_{0.5},\hat q_{0.9}]
$$

训练时采用 pinball quantile loss，并叠加单调惩罚：

$$
\mathcal L=
\frac{1}{NHTQ}\sum \max(q\cdot e,(q-1)\cdot e)
+ \lambda\Big(\mathrm{ReLU}(q_{10}-q_{50})+\mathrm{ReLU}(q_{50}-q_{90})\Big)
$$

代码里：
- `Q = {0.1, 0.5, 0.9}`
- `λ = 0.1`

推理时还会强制排序，确保：

$$
q_{10}\le q_{50}\le q_{90}
$$

## 6. Baseline 现在是四条

当前 baseline 固定为：

- `persistence`
- `linear_trend`
- `seasonal_naive`
- `harmonic_trend`

它们都会参与训练摘要、评估和对比 CSV。

## 7. 正式不确定性：`cqr_conformal_v1`

当前不再把启发式区间当正式输出，正式不确定性模式是：

$$
uncertainty\_mode = cqr\_conformal\_v1
$$

在验证集上，对每个 horizon 计算：

$$
s_h^{(n)}=\max(q_{10,raw}^{(n)}-y^{(n)},\ y^{(n)}-q_{90,raw}^{(n)},\ 0)
$$

再取 conformal 修正量 $\delta_h$：

$$
q_{10,cal}=q_{10,raw}-\delta_h,\qquad
q_{90,cal}=q_{90,raw}+\delta_h
$$

目标覆盖率固定为：

$$
target\_coverage = 0.80
$$

## 8. `c_pred` 现在是什么

`c_pred` 不再是“模型输出的置信度”，而是后处理得到的 risk-aware confidence surrogate：

$$
width\_norm=\mathrm{norm}(interval\_width_{cal})
$$

$$
c_{pred}=reliability\cdot\Big[
0.60(1-width\_norm)+
0.20\,ps\_score+
0.10(1-jump\_risk)+
0.10(1-anomaly\_exposure)
\Big]
$$

其中 `reliability` 来自验证集 calibration summary。

## 9. 输出 schema

### 9.1 训练摘要

- `forecast_train_summary.json`
- `forecast_data_summary.json`
- `forecast_calibration.json`
- `forecast_explainability.json`

关键字段：

- `forecast_mode_requested`
- `forecast_mode_actual`
- `active_model`
- `generic_model_name`
- `generic_model_selection`
- `selection_mode`
- `zone_filter_mode`
- `graph_stats`
- `fallback_triggered / fallback_reasons`

### 9.2 推理 HDF5

`forecast_predictions.h5` 中：

- `meta/forecast_mode_requested`
- `meta/forecast_mode_actual`
- `meta/uncertainty_mode`
- `meta/active_model`
- `meta/history_dates`
- `meta/future_dates`
- `points/*`
- `predictions/pred_offset_p10|p50|p90`：校准后
- `predictions/pred_rel0_p10|p50|p90`：校准后
- `predictions/*_raw`：原始分位数
- `predictions/interval_width_raw`
- `predictions/interval_width_calibrated`
- `predictions/c_pred`

### 9.3 评估与图件

评估会统一比较：

- 4 条 baseline
- `generic`
- `hazard`（若训练成功）
- `active_model`

图件包括：

- `forecast_confidence_map.png`
- `forecast_horizon_panel.png`
- `forecast_calibration_curve.png`
- `forecast_feature_group_importance.png`
- `forecast_neighbor_influence_map.png`（hazard 有效时）

## 10. 一句话总结

当前 MintPy 后链已经变成：

> 共享分解数据层 + generic/hazard 双模式 + baseline 对照 + conformal 校准 + explainability 的正式预测链

而不是旧版“GRU 单模型 + 启发式置信度”。
