# Git仓库信息

**创建时间**: 2026-02-09 20:01  
**仓库位置**: `/hy-tmp/paddle_detection_mouse/Mobile_Deployment/`

---

## 📦 Git提交信息

### 提交哈希
```
6b920f147b9b78000b3d30a4b0db9b09cdb4cc96
```

### 提交信息
```
feat: 完成模型量化和ONNX转换，准备移动端部署

- 导出YOLOv3模型为Paddle Inference格式
- 使用PTQ离线量化压缩模型
- 转换FP32模型为ONNX格式 (92.34 MB)
- 添加完整的部署文档和Mac开发指南
- 包含模型文件、配置文件和标签文件

模型信息:
- 架构: YOLOv3 + MobileNetV1
- 任务: 实验鼠检测 (mouse/other)
- 精度: mAP@0.5 = 93.63%
- 输入: 608x608x3 RGB
- 格式: ONNX Opset 11

交付物:
- yolov3_mouse_fp32.onnx (92.34 MB)
- infer_cfg.yml, label_list.txt
- 完整开发文档 (README, MAC_SETUP_GUIDE等)
```

---

## 📊 仓库统计

### 文件清单
```
12 files changed, 2002 insertions(+)

- .gitignore                    (11 行)
- CHECKLIST.md                  (235 行)
- LINUX_WORK_SUMMARY.md         (314 行)
- MAC_SETUP_GUIDE.md            (470 行)
- QUICK_REFERENCE.md            (210 行)
- README.md                     (499 行)
- START_HERE.txt                (22 行)
- models/infer_cfg.yml          (27 行)
- models/label_list.txt         (2 行)
- models/model_info.md          (212 行)
- models/yolov3_mouse_fp32.onnx (96,825,091 字节 = 92.34 MB)
- progress_tracker.md           (0 行)
```

### 仓库大小
- **.git目录**: 87 MB
- **工作目录**: 93 MB
- **总计**: ~180 MB

### Git对象统计
- **对象数量**: 15
- **对象大小**: 86.11 MiB
- **打包文件**: 0

---

## ✅ 已纳入Git管理的文件

### 模型文件 ✓
- [x] `models/yolov3_mouse_fp32.onnx` (92.34 MB) - **已包含，未忽略**

### 配置文件 ✓
- [x] `models/infer_cfg.yml`
- [x] `models/label_list.txt`

### 文档文件 ✓
- [x] `README.md`
- [x] `MAC_SETUP_GUIDE.md`
- [x] `LINUX_WORK_SUMMARY.md`
- [x] `QUICK_REFERENCE.md`
- [x] `CHECKLIST.md`
- [x] `models/model_info.md`
- [x] `START_HERE.txt`

### 其他文件 ✓
- [x] `.gitignore`
- [x] `progress_tracker.md`

---

## 🔧 Git配置

### 用户信息
```
user.name: Mobile Deployment
user.email: deployment@paddledetection.local
```

### .gitignore规则
```gitignore
# 临时文件
*.log
*.tmp
*~
.DS_Store

# 编辑器
.vscode/
.idea/

# 但是保留模型文件（不忽略.onnx）
```

**注意**: `.onnx`文件**不会**被忽略，模型文件已完整提交到Git仓库。

---

## 📋 Git常用命令

### 查看提交历史
```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment
git log --oneline
git log --stat
```

### 查看文件状态
```bash
git status
git ls-files
```

### 查看特定文件
```bash
git log -- models/yolov3_mouse_fp32.onnx
git show HEAD:models/yolov3_mouse_fp32.onnx
```

### 创建标签
```bash
git tag -a v1.0 -m "Release v1.0: 初始移动端部署版本"
git tag -l
```

---

## 🚀 推送到远程仓库（可选）

如果需要推送到远程Git仓库（如GitHub、GitLab）：

### 1. 添加远程仓库
```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment
git remote add origin <your-repo-url>
```

### 2. 推送代码
```bash
git push -u origin master
```

### ⚠️ 注意事项
由于模型文件较大（92.34 MB），推送到远程仓库时需要注意：

1. **GitHub限制**: 单个文件不能超过100MB（当前92.34MB可以）
2. **GitLab限制**: 默认单个文件不能超过100MB
3. **推荐使用Git LFS**: 对于大文件，建议使用Git Large File Storage

### 使用Git LFS（推荐）
```bash
# 安装Git LFS
git lfs install

# 追踪ONNX文件
git lfs track "*.onnx"

# 添加.gitattributes
git add .gitattributes

# 提交并推送
git commit -m "chore: 配置Git LFS追踪ONNX文件"
git push -u origin master
```

---

## 📦 克隆仓库

在Mac上克隆此仓库：

```bash
# 如果使用本地路径
git clone /path/to/Mobile_Deployment ~/Projects/MouseDetection_Mobile

# 如果推送到远程仓库
git clone <your-repo-url> ~/Projects/MouseDetection_Mobile
```

---

## ✅ 验证

### 确认模型文件已提交
```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment
git ls-files models/yolov3_mouse_fp32.onnx
# 输出: models/yolov3_mouse_fp32.onnx

ls -lh models/yolov3_mouse_fp32.onnx
# 输出: -rw-r--r-- 1 root root 93M Feb  9 19:52 models/yolov3_mouse_fp32.onnx
```

### 确认所有文件已提交
```bash
git status
# 输出: On branch master
#       nothing to commit, working tree clean
```

---

## 📝 更新日志

### v1.0 (2026-02-09)
- ✅ 初始提交
- ✅ 包含ONNX模型文件（92.34 MB）
- ✅ 包含完整部署文档
- ✅ 包含Mac开发指南

---

**Git仓库已成功创建并提交所有文件（包括模型文件）！** ✅

**最后更新**: 2026-02-09 20:05

