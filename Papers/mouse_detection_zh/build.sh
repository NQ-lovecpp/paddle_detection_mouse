#!/bin/bash
# 完整编译流程：目录与参考文献需多次运行才能正确生成
# 1. 首次 xelatex：生成 .aux, .toc
# 2. bibtex：处理参考文献
# 3. 再次 xelatex：纳入 .bbl
# 4. 末次 xelatex：固定交叉引用与目录
cd "$(dirname "$0")"
set -e
echo "=== 第 1/4 步：xelatex (初编) ==="
xelatex -interaction=nonstopmode main.tex >/dev/null 2>&1
echo "=== 第 2/4 步：bibtex (参考文献) ==="
bibtex main || true
echo "=== 第 3/4 步：xelatex (纳入 .bbl) ==="
xelatex -interaction=nonstopmode main.tex >/dev/null 2>&1
echo "=== 第 4/4 步：xelatex (固定引用) ==="
xelatex -interaction=nonstopmode main.tex
echo "=== 完成：main.pdf 已生成 ==="
