# Git远程推送指南

**问题**: 模型文件(92MB)需要推送到远程仓库

---

## 🎯 推荐方案

### 方案1: 使用Git LFS（推荐，适合大文件）

#### 1.1 安装Git LFS
```bash
# 检查是否已安装
git lfs version

# 如果未安装，安装Git LFS
# Ubuntu/Debian
apt-get install git-lfs

# 或从源码安装
wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.0/git-lfs-linux-amd64-v3.4.0.tar.gz
tar -xzf git-lfs-linux-amd64-v3.4.0.tar.gz
cd git-lfs-3.4.0
sudo ./install.sh
```

#### 1.2 配置Git LFS
```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment

# 初始化Git LFS
git lfs install

# 追踪ONNX文件
git lfs track "*.onnx"

# 添加.gitattributes
git add .gitattributes
git commit -m "chore: 配置Git LFS追踪ONNX文件"

# 迁移现有的ONNX文件到LFS
git lfs migrate import --include="*.onnx" --everything
```

#### 1.3 推送到远程
```bash
# 添加远程仓库
git remote add origin <your-repo-url>

# 推送（LFS会自动处理大文件）
git push -u origin master
```

---

### 方案2: 直接推送（适合GitHub/GitLab，文件<100MB）

当前模型文件92.34MB，在GitHub的100MB限制内，可以直接推送。

```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment

# 添加远程仓库（替换为你的实际地址）
git remote add origin https://github.com/your-username/your-repo.git
# 或使用SSH
git remote add origin git@github.com:your-username/your-repo.git

# 推送到远程
git push -u origin master
```

**注意**: 
- GitHub单文件限制: 100MB（当前92.34MB ✓ 可以）
- GitLab单文件限制: 默认100MB
- 如果推送失败，使用方案1（Git LFS）

---

### 方案3: 分离模型文件（不推荐）

如果不想将模型文件放入Git：

```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment

# 从Git中移除模型文件
git rm --cached models/yolov3_mouse_fp32.onnx

# 添加到.gitignore
echo "models/*.onnx" >> .gitignore

# 提交更改
git add .gitignore
git commit -m "chore: 从Git中移除ONNX模型文件"

# 推送
git remote add origin <your-repo-url>
git push -u origin master
```

然后单独上传模型文件到：
- 云存储（Google Drive, Dropbox, 百度网盘）
- GitHub Releases
- 对象存储（AWS S3, 阿里云OSS）

---

## 📝 具体操作步骤

### 如果你有GitHub仓库

1. **在GitHub创建新仓库**
   - 访问 https://github.com/new
   - 仓库名: `mouse-detection-mobile`
   - 不要初始化README（因为本地已有）

2. **推送代码**
```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment

# 添加远程仓库（替换your-username）
git remote add origin https://github.com/your-username/mouse-detection-mobile.git

# 推送
git push -u origin master
```

3. **如果推送失败（文件太大）**
```bash
# 使用Git LFS（参考方案1）
git lfs install
git lfs track "*.onnx"
git add .gitattributes
git commit -m "chore: 配置Git LFS"
git lfs migrate import --include="*.onnx" --everything
git push -u origin master --force
```

---

### 如果你使用GitLab

```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment

# 添加GitLab远程仓库
git remote add origin https://gitlab.com/your-username/mouse-detection-mobile.git

# 推送
git push -u origin master
```

---

### 如果你使用自建Git服务器

```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment

# 添加远程仓库
git remote add origin user@your-server:/path/to/repo.git

# 推送
git push -u origin master
```

---

## 🔍 验证推送

推送成功后，验证：

```bash
# 查看远程仓库
git remote -v

# 查看远程分支
git branch -r

# 查看推送状态
git log --oneline origin/master
```

在远程仓库网页上检查：
- [ ] 所有文档文件已上传
- [ ] models/yolov3_mouse_fp32.onnx 已上传（或在LFS中）
- [ ] 提交历史完整

---

## ⚠️ 常见问题

### Q1: 推送时提示文件太大
```
remote: error: File models/yolov3_mouse_fp32.onnx is 92.34 MB; this exceeds GitHub's file size limit of 100 MB
```

**解决**: 使用Git LFS（方案1）

### Q2: 推送速度很慢
**原因**: 92MB文件上传需要时间

**解决**: 
- 使用更快的网络
- 或使用Git LFS（只上传一次）
- 或使用方案3（分离模型文件）

### Q3: 推送被拒绝
```
! [rejected]        master -> master (fetch first)
```

**解决**:
```bash
git pull origin master --rebase
git push -u origin master
```

---

## 💡 推荐做法

**对于这个项目，我推荐**:

1. **如果文件<100MB**: 直接推送（方案2）✅ 当前92.34MB可以
2. **如果文件>100MB**: 使用Git LFS（方案1）
3. **如果经常更新模型**: 使用Git LFS + GitHub Releases

---

## 🚀 快速命令（直接推送）

```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment

# 1. 添加远程仓库（替换为你的地址）
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 2. 推送
git push -u origin master

# 3. 验证
git remote -v
```

---

**需要我帮你执行哪个方案？请提供你的远程仓库地址。**

