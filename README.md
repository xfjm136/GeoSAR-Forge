# GeoSAR-Forge

GeoSAR-Forge is a Sentinel-1 PS/DS-InSAR processing and forecasting project for automated ground-deformation monitoring.  
It combines `ISCE2`, `Dolphin`, `Mainchain QC / DePSI-like QC`, `MintPy`, `support_graph_v1` deformation-zone detection, and `mintpy_forecast` v2 high-confidence point forecasting in one workflow.

---

## 中文简介

GeoSAR-Forge 是一个面向 Sentinel-1 多时相数据的 PS/DS-InSAR 地表形变自动化处理项目。  
当前正式主线是：

`ISCE2 → Dolphin → 主链 QC / DePSI-like QC → MintPy(pass1→feedback→pass2) → support_graph_v1 区域识别 → mintpy_forecast v2`

### 核心能力

- 主 InSAR 反演：从 SLC 下载、精轨、ISCE2 配准到 Dolphin 相位连接和 MintPy 两阶段时序反演
- 主链质量控制：pair 级 QC、DePSI-like 高可信点骨架、参考网络和异常日期反馈
- 区域识别：`support_graph_v1` 在高可信点图上做异常形变区分割，并输出 polygon 与区域累计位移曲线
- 未来预测：`mintpy_forecast` v2 提供 `generic / hazard` 双模式、多基线对照和 conformal 校准区间
- 工程化输出：GeoTIFF、Shapefile、KMZ、CSV、HDF5、PNG/PDF 图件与 notebook 工作流

### 仓库结构

```text
.
├── pipeline.ipynb              # 主 notebook 入口
├── docs/                       # 中文技术文档、流程图与说明图
├── insar_utils/                # 主链、QC、导出与区域识别模块
├── mintpy_forecast/            # 当前正式预测包
├── setup_env.sh                # 环境搭建脚本
├── sitecustomize.py            # ISCE2 兼容补丁
├── .env.example                # 环境变量模板
└── README.md
```

### 快速开始

1. 创建环境

```bash
bash setup_env.sh
```

2. 配置敏感信息

```bash
cp .env.example .env
```

然后在 `.env` 中填写：

- `ASF_USER`, `ASF_PASS`
- `ESA_USER`, `ESA_PASS`
- `OPENTOPO_KEY`
- `ERA5_URL`, `ERA5_KEY`

3. 启动 notebook

主入口是 [pipeline.ipynb](./pipeline.ipynb)。

### 文档入口

- [当前正式流程技术文档](./docs/PIPELINE.md)
- [用户手册](./docs/用户手册.md)
- [DePSI-like QC 创新说明](./docs/DePSI_like_QC_创新说明.md)
- [MintPy 后未来预测说明](./docs/MintPy后高可信点预测说明.md)


---

## English Overview

GeoSAR-Forge is a Sentinel-1 PS/DS-InSAR workflow for automated ground deformation monitoring.  
The current production chain is:

`ISCE2 → Dolphin → Mainchain QC / DePSI-like QC → MintPy(pass1→feedback→pass2) → support_graph_v1 zone detection → mintpy_forecast v2`

### Main Features

- End-to-end InSAR processing from scene selection and download to MintPy time-series inversion
- Quality-aware pipeline with pair-level QC, high-confidence PS masks, reference network construction, and anomaly-date feedback
- Graph-based deformation-zone detection with polygon export and regional cumulative displacement curves
- Forecasting with `generic / hazard` modes, baseline comparison, and conformal uncertainty calibration
- Engineering-friendly outputs in GeoTIFF, Shapefile, KMZ, CSV, HDF5, PNG, and PDF

### Repository Layout

```text
.
├── pipeline.ipynb
├── docs/
├── insar_utils/
├── mintpy_forecast/
├── setup_env.sh
├── sitecustomize.py
├── .env.example
└── README.md
```

### Quick Start

1. Build the environment

```bash
bash setup_env.sh
```

2. Create your local secrets file

```bash
cp .env.example .env
```

3. Fill in the required credentials in `.env`, then open [pipeline.ipynb](./pipeline.ipynb).

### Documentation

- [Pipeline documentation](./docs/PIPELINE.md)
- [User guide](./docs/用户手册.md)
- [DePSI-like QC notes](./docs/DePSI_like_QC_创新说明.md)
- [MintPy forecast notes](./docs/MintPy后高可信点预测说明.md)
