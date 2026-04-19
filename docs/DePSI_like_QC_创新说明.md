# GeoSAR-Forge 相对传统 ISCE2 + Dolphin + MintPy 工作流的创新说明

这份文档专门回答一个问题：相比传统的 `ISCE2 + Dolphin + MintPy` 工作流，当前 GeoSAR-Forge 具体做了哪些创新，这些创新改动了哪些数据流、加入了哪些公式、以及它们为什么能提高主链的可控性和解释性。

需要先说明的是，我们当前并没有把 DePSI 软件接入主链，也没有把处理流程改成“另一套 PSI 软件替代 MintPy”。真正做的事情是：**把更偏 PSI / DePSI 的质量控制思想前移到 MintPy 之前**，让高可信 PS 骨架、时间模型一致性、参考主网络和异常日期反馈直接参与主链，而不是等最终时序出来后再被动诊断。并且这些产物现在还会继续向下游提供给：

- `support_graph_v1` 异常形变区识别
- MintPy 后未来预测 v2

## 1. 从传统主链到当前主链

传统链路可以概括成：

$$
\text{ISCE2} \rightarrow \text{Dolphin} \rightarrow \text{MintPy}
$$

当前主链则改成：

$$
\text{ISCE2} \rightarrow \text{Dolphin} \rightarrow \text{Mainchain QC / DePSI-like QC} \rightarrow \text{MintPy}
$$

其中新增的 `Mainchain QC / DePSI-like QC` 会显式产生：

- `pair_qc.csv`
- `ifgramStack_qc.h5`
- `ps_score.tif`
- `mask_ps_strict.tif`
- `mask_ps_relaxed.tif`
- `ref_primary_network.csv`
- `ref_candidates.csv`
- `date_qc.csv`

这意味着我们不再把所有干涉对和所有可用像元平权地送进 MintPy，而是先构建一个更保守、更高可信的输入骨架。

## 2. 创新一：pair 级质量控制前移

传统流程里，坏 pair 常常要等 MintPy 最终时序跳变之后才被动暴露。当前实现把它前移到了 MintPy 之前。

我们先对每对干涉图计算：

- `coh_mean`
- `valid_conncomp_frac`
- `n_conncomp`
- `pair_p95_abs_los_mm`
- `pair_frac_abs_los_gt_50mm`

并用如下风险分数进行排序：

$$
risk = 0.25(1-\mathrm{norm\_coh})
      + 0.20(1-\mathrm{valid\_conncomp\_frac})
      + 0.15\mathrm{norm}(n\_conncomp)
      + 0.20\mathrm{norm}(p95\_abs\_los)
      + 0.20\mathrm{norm}(frac_{>50mm})
$$

随后把 pair 划分为：

- `keep`
- `downweight`
- `drop`

并单独生成 `ifgramStack_qc.h5`。这里的创新点不是“又做了一次阈值筛选”，而是把质量控制显式变成了 **主链输入网络的一部分**。这样 MintPy 接收到的 stack 已经是经过保守整理的版本，而不是原始 stack 的无差别输入。

## 3. 创新二：多指标 `ps_score` 替代单一相干阈值

传统工作流更容易依赖单一的 `temporal coherence` 阈值来判断点是否可用。这样做简单，但不能真正回答“哪些点是高可信骨架、哪些点只是能算出来”。

当前我们定义了一个更保守的点级评分：

$$
ps\_score =
0.30\,tcoh +
0.20\,valid\_pair\_ratio +
0.20\,mainCC\_ratio +
0.15(1-model\_rms\_norm) +
0.15(1-jump\_risk)
$$

这里每一项都对应一个具体质量来源：

- `tcoh`：Dolphin temporal coherence
- `valid_pair_ratio`：该点在所有候选 pair 中有效相位的比例
- `mainCC_ratio`：该点位于 pair 主连通分量中的比例
- `model_rms_norm`：时间模型残差的鲁棒归一化
- `jump_risk`：该点是否被少数异常 pair 主导

为了便于排错，系统还额外输出分量图：

- `tcoh_component.tif`
- `mainCC_component.tif`
- `model_component.tif`
- `jump_component.tif`

这一步真正改变的是“高可信”的定义方式：从“相干性高就行”，变成“相干性高、主连通分量稳定、时间模型可解释、且不被异常 pair 主导，才算高可信”。

## 4. 创新三：strict / relaxed 分层与覆盖率告警

在 `ps_score` 基础上，系统继续定义两层掩膜：

$$
M_{\mathrm{strict}}(i)=\mathbf{1}\big(ps\_score_i \ge \tau_{\mathrm{strict}}\big)
$$

$$
M_{\mathrm{relaxed}}(i)=\mathbf{1}\big(\tau_{\mathrm{relaxed}} \le ps\_score_i < \tau_{\mathrm{strict}}\big)
$$

默认阈值是：

$$
\tau_{\mathrm{strict}} = 0.75,\qquad \tau_{\mathrm{relaxed}} = 0.60
$$

但这里不把它们当成固定常数，而是同步记录：

$$
strict\_coverage\_ratio = \frac{\sum_i M_{\mathrm{strict}}(i)}{N_{\mathrm{valid}}}
$$

$$
relaxed\_coverage\_ratio = \frac{\sum_i M_{\mathrm{relaxed}}(i)}{N_{\mathrm{valid}}}
$$

以及：

- `strict_candidate_count`
- `coverage_warning_flag`
- `coverage_warning_reason`

其意义是：如果 strict 覆盖过低，说明当前项目上阈值可能过严；如果 strict 或 strict+relaxed 覆盖过高，则说明区分度不够。这样我们不再只是“输出两个 mask”，而是同时输出这些 mask 是否合理的摘要信息。

## 5. 创新四：时间模型一致性进入主链

这一步最接近 PSI / DePSI 的思想。传统链路里，一个点只要相干性不错，就容易继续往下走；但在 PSI 思路里，高质量点还必须“时间行为可解释”。

当前在 candidate 点上拟合 3 类轻量模型：

1. 线性
2. 线性 + 年周期
3. 单拐点分段线性

然后按 `BIC` 选择最佳模型，并输出：

- `best_model`
- `model_rms`
- `delta_bic`
- `leave_one_pair_out_sensitivity`

这里最关键的量是：

$$
model\_rms = \sqrt{\frac{1}{T}\sum_t (x_t-\hat x_t)^2}
$$

它反映了该点时序能否被合理模型解释。`delta_bic` 则描述最佳模型和第二佳模型之间的差距；`leave_one_pair_out_sensitivity` 反映某个点是否过度依赖少数 pair。  

这一步的创新点，是把“时间模型一致性”从一个后验诊断指标，变成了主链里真正参与 `ps_score`、参考点排序和反馈判定的核心量。

## 6. 创新五：自适应 jump 判据

如果一开始就把 jump 阈值写死为某个常数，不同项目之间很容易失真。因此当前实现改为“默认值 + 数据自适应”。

默认基准阈值为：

$$
jump\_threshold\_{base} = 28\ \mathrm{mm}
$$

同时在 strict PS 上统计 residual 分布，并采用：

$$
jump\_threshold\_{applied}
=
\max\left(
jump\_threshold\_{base},
P95(residual)\times jump\_threshold\_{scale}
\right)
$$

默认：

$$
jump\_threshold\_{scale}=1.0
$$

然后定义：

$$
jump\_risk = \frac{N(|residual| > jump\_threshold\_{applied})}{N_{\mathrm{valid\ pair}}}
$$

这样，当某个项目本身 residual 尺度偏大时，系统不会轻易把大量强但合理的变化误判成 jump；反过来，如果 residual 整体较小，仍然会保留默认 28 mm 作为底线。

## 7. 创新六：参考点不再是“最高相干像元”

传统工作流里，参考点常常会退化成“自动沿用已有 REF_X/REF_Y”或者“挑最高相干点”。当前系统改成了三层逻辑：

第一层，在 strict PS 内建立 **稀疏参考主网络** `ref_primary_network`。  
第二层，从这个网络里生成 `top N` 候选。  
第三层，为每个候选统计一个小参考区，而不是只看单个像元。

当前参考点核心评分为：

$$
ref\_score =
0.35\,tcoh +
0.25\,mainCC\_ratio +
0.20(1-model\_rms\_norm) +
0.10\,valid\_pair\_ratio +
0.10\,neighbor\_stability
$$

而在参考主网络排序时，又进一步加入：

- `dist_to_edge`
- `strict_neighbor_count`
- `local_velocity_gradient`
- `gacos_safe_flag`

如果对候选点 $i$ 周围半径 $r$ 的 strict patch 记为 $\mathcal{P}_i(r)$，则 patch 级统计量写成：

$$
reference\_patch\_ps\_score\_median(i)=
\mathrm{median}\{ps\_score_j\mid j\in \mathcal{P}_i(r)\cap M_{\mathrm{strict}}\}
$$

$$
reference\_patch\_model\_rms\_median(i)=
\mathrm{median}\{model\_rms_j\mid j\in \mathcal{P}_i(r)\cap M_{\mathrm{strict}}\}
$$

$$
reference\_patch\_jump\_risk\_median(i)=
\mathrm{median}\{jump\_risk_j\mid j\in \mathcal{P}_i(r)\cap M_{\mathrm{strict}}\}
$$

因此，当前参考点虽然最终还是一个像元坐标，但它代表的是“位于稳定小参考区中心的候选”，而不是“一个单像元极值点”。这一步直接提高了参考点选择的稳健性。

## 8. 创新七：异常日期区分全局和局部

传统链路里，一个日期异常通常只会体现在“残差偏大”。当前我们把它继续拆分成“全局异常”和“局部异常”。

在 strict PS 上，对每个日期统计：

- `median_abs_residual_mm`
- `frac_gt_threshold`
- `signed_bias_mm`

同时把异常点按粗网格统计集中程度，定义：

$$
spatial\_concentration\_index =
\frac{\max_g N_g(\mathrm{abnormal})}{N(\mathrm{abnormal})}
$$

其中 $N_g(\mathrm{abnormal})$ 表示某个空间粗网格内的异常点数。  

据此定义：

- `global`：空间上分散，更像大气、轨道或整体几何问题
- `mixed`
- `local`：空间上高度集中，更像局部解缠或局部边界问题

这一步的创新点在于，异常日期不再只是一维残差指标，而带上了空间结构信息，从而可以更细地回灌到 pair 级 QC。

## 9. 创新八：两阶段 MintPy 反馈回灌

当前 MintPy 不再只跑一轮，而是：

$$
pass1 \rightarrow feedback \rightarrow pass2
$$

系统先在 `pass1` 结果上做 strict PS 群体残差统计，再把异常日期映射回 incident pairs，形成：

$$
q\_{pair}^{(2)} = \max\big(0,\ q\_{pair}^{(1)} - feedback\_penalty\big)
$$

而 `feedback_penalty` 会根据异常日期的作用范围分层累计：

- `global` 日期惩罚更大
- `mixed` 次之
- `local` 最小

并受 `strict_ps_consistency` 等条件进一步调节。

这一机制的意义，不是追求无限迭代，而是在保持主链可解释的前提下，让高可信骨架反过来指导网络更新。

## 10. 创新九：双成果输出

传统工作流通常默认只输出一套结果。当前系统明确区分：

- `full_coverage`
- `high_confidence`

也就是说：

- 高可信骨架决定可信度
- 普通有效像元决定覆盖度

这一步把“结果能不能看”和“结果值不值得信”从同一层里拆了出来，更适合工程使用和汇报表达。

## 11. 与下游区域识别和未来预测的衔接

当前系统已经不再在 MintPy 后做深度学习时序矫正，也不再把 DePSI-like QC 只当成主链内部的质量控制。当前真实后链是：

$$
\text{MintPy accepted result}
\rightarrow
\texttt{support\_graph\_v1 zone detection}
\rightarrow
\texttt{forecast v2}
$$

这里 DePSI-like QC 的创新，直接成为两个下游模块的输入来源：

- `ps_score`
- `tcoh`
- `valid_pair_ratio`
- `mainCC_ratio`
- `model_rms`
- `jump_risk`
- `anomaly_exposure`

对区域识别链，这些字段决定：

- `strict ∪ relaxed` 支持点集合
- `qc_support` 和高可信骨架密度
- 区域边界 outward growth 的可信约束

对预测链，这些字段决定：

- `selection_mode` 的 strict / relaxed 选点策略
- 8 维静态特征中的 `ps_score / tcoh / valid_pair_ratio / mainCC_ratio / jump_risk / anomaly_exposure`
- `c_pred` 风险感知置信代理量中的质量先验

也就是说，DePSI-like QC 的价值不只体现在“主链更稳”，还体现在它把质量信息继续传给了 **区域识别** 和 **未来预测** 两条后链，从而让 MintPy 后处理从“结果矫正”转成了“高可信区域识别 + 高可信点未来预测”。

## 12. 一句话总结

相对传统 `ISCE2 + Dolphin + MintPy` 工作流，当前 GeoSAR-Forge 的创新不在于换掉哪一个软件，而在于：**把高可信 PS 骨架、时间模型一致性、参考主网络和异常日期反馈前移到 MintPy 之前，并把这些质量信息继续传给 MintPy 后的区域识别链和未来预测链。**
