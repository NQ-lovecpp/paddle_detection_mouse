---
name: LaTeX论文工程
overview: 在工作区新建 Papers/ 目录，完整搭建 LaTeX 环境（TUNA 镜像加速），撰写一篇基于 8 组真实训练数据的技术论文，覆盖数据工程、模型训练、部署全流程，使用 pgfplots/TikZ 绘制精确图表。
todos:
  - id: install-latex
    content: 配置 TUNA apt 镜像并安装 texlive-latex-extra + texlive-science
    status: completed
  - id: create-structure
    content: 新建 Papers/mouse_detection/ 目录结构，创建 main.tex 和各 section 文件骨架
    status: completed
  - id: write-sections-1-4
    content: 撰写 Introduction / Dataset / Methodology / Experimental Design 四章
    status: completed
  - id: write-figures
    content: 用 pgfplots 编写 Fig.1-5（L1 mAP 曲线、YOLOv3 曲线、消融柱状图、散点气泡图）和 TikZ 流程图
    status: completed
  - id: write-tables
    content: 用 booktabs 编写三张汇总表（超参配置、实验结果、模型规格对比）
    status: completed
  - id: write-sections-5-8
    content: 撰写 Results / Deployment / Future Work / Conclusion 四章
    status: completed
  - id: write-references
    content: 编写 references.bib，收录关键文献
    status: completed
  - id: compile-pdf
    content: 运行 latexmk 编译并修复所有 warning/error，产出最终 PDF
    status: completed
isProject: false
---

# LaTeX 技术论文工程

## 目标

撰写一篇完整的学术技术报告，工程目录为 `Papers/mouse_detection/`，最终生成 `main.pdf`。

---

## Step 1：LaTeX 环境安装

使用清华 TUNA apt 镜像（国内速度最快）：

```bash
# 替换 apt 源为 TUNA
sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
apt-get update
apt-get install -y texlive-latex-extra texlive-fonts-recommended \
    texlive-science texlive-bibtex-extra latexmk
```

所需包：`pgfplots`（图表）、`tikz`（流程图）、`booktabs`（三线表）、`algorithm2e`（伪代码）、`listings`（代码块）、`hyperref`、`cleveref`。

---

## Step 2：论文结构

**文件布局**：

```
Papers/mouse_detection/
├── main.tex          # 主文件（IEEE/ACM 单栏格式）
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── dataset.tex
│   ├── methodology.tex
│   ├── experiments.tex
│   ├── deployment.tex
│   ├── future_work.tex
│   └── conclusion.tex
├── figures/          # TikZ/pgfplots 内联图表
└── references.bib
```

**章节大纲**：

- Abstract
- 1. Introduction（背景、问题定义、贡献列表）
- 1. Dataset & Data Engineering（3 源整合、VOC 格式、7,286 张统计、1/3 子集设计）
- 1. Model Architectures（PicoDet-S LCNet 骨干、YOLOv3-MobileNetV1、参数对比表）
- 1. Experimental Design（Linear Scaling Rule 推导、4×2 矩阵、超参配置表）
- 1. Results & Analysis
  - 5.1 PicoDet-S 消融（L1–L4，mAP 曲线 + 柱状图）
  - 5.2 YOLOv3 消融（Y1–Y4，数据量效应）
  - 5.3 跨模型对比（精度-速度-尺寸三维散点图）
- 1. Mobile Deployment（导出链路图、iOS 四轮迭代、1fps→14fps）
- 1. Future Work（A/B/C 三组：模型扩展、Anchor 聚类、压缩流水线）
- 1. Conclusion

---

## Step 3：图表计划（全部基于真实数据）


| 图编号   | 类型          | 数据来源                               | 内容                                                                   |
| ----- | ----------- | ---------------------------------- | -------------------------------------------------------------------- |
| Fig.1 | pgfplots 折线 | `L1 TRAINING_REPORT.md` 逐 epoch 数据 | L1 [mAP@0.5](mailto:mAP@0.5) 曲线（epoch 10→299），标注峰值 epoch 70 = 94.30% |
| Fig.2 | pgfplots 折线 | `Y3 train.log` grep 结果             | YOLOv3-Y3 mAP 曲线（epoch 5→80），对比 PicoDet 收敛速度                         |
| Fig.3 | pgfplots 柱状 | `lightweight_summary.csv` + L1 数据  | L1–L4 mAP 对比（配色区分 bs/lr/epoch 变量）                                    |
| Fig.4 | pgfplots 柱状 | `yolov3_summary.csv`               | Y1–Y4 mAP 对比，叠加数据量轴，凸显 +47pp 效应                                      |
| Fig.5 | pgfplots 散点 | 全 8 组汇总                            | mAP vs 模型尺寸（mb）+ FPS 气泡图                                             |
| Fig.6 | TikZ 流程图    | —                                  | 训练→导出→ONNX→iOS 部署链路                                                  |
| Fig.7 | pgfplots 折线 | CHECKLIST.md 迭代记录                  | iOS 推理速度四轮迭代折线（1→4→8→14fps）                                          |
| Tab.1 | booktabs 表  | 全部                                 | 8 组实验超参配置总表                                                          |
| Tab.2 | booktabs 表  | 全部                                 | 8 组实验结果汇总表（最终对比）                                                     |
| Tab.3 | booktabs 表  | CHECKLIST.md                       | 模型规格对比（PicoDet-S vs YOLOv3）                                          |


---

## Step 4：编译

```bash
cd Papers/mouse_detection
latexmk -pdf -interaction=nonstopmode main.tex
```

若 pgfplots 渲染慢，使用 `\usepgfplotslibrary{external}` 启用图表缓存。

---

## 关键技术决策

- **论文格式**：`\documentclass[12pt,a4paper]{article}`，单栏，适合技术报告（比 IEEE 双栏更适合图表展示）
- **中文支持**：暂不引入，全英文撰写，避免 XeLaTeX + 中文字体的安装复杂度
- **图表渲染**：全部使用 pgfplots/TikZ 原生 LaTeX 代码（无外部图片依赖，便于版本控制和精确调整）
- **参考文献**：BibTeX，收录 PicoDet、YOLOv3、Linear Scaling Rule 等关键文献

