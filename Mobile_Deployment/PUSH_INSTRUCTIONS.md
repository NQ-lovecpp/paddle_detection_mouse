# 推送到GitHub详细步骤

**远程仓库**: https://github.com/NQ-lovecpp/paddle_detection_mouse.git  
**当前状态**: 远程仓库已配置，需要身份验证

---

## 🔐 方式1: 使用Personal Access Token（推荐）

### 步骤1: 创建GitHub Token

1. 访问: https://github.com/settings/tokens/new
2. 填写信息:
   - **Note**: `paddle_detection_mouse_deploy`
   - **Expiration**: 选择有效期（建议90天或自定义）
   - **Select scopes**: 勾选 `repo` (完整仓库权限)
3. 点击 **Generate token**
4. **重要**: 复制生成的token（只显示一次！）

### 步骤2: 推送代码

```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment

# 推送
git push -u origin master

# 输入凭据:
# Username: NQ-lovecpp
# Password: <粘贴你的token>
```

### 步骤3: 保存凭据（可选）

```bash
# 缓存凭据15分钟
git config --global credential.helper cache

# 或永久保存（不推荐）
git config --global credential.helper store
```

---

## 🔐 方式2: 使用SSH密钥

### 步骤1: 生成SSH密钥

```bash
# 生成密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 按Enter使用默认路径
# 可以设置密码或直接Enter跳过

# 查看公钥
cat ~/.ssh/id_ed25519.pub
```

### 步骤2: 添加到GitHub

1. 复制公钥内容
2. 访问: https://github.com/settings/keys
3. 点击 **New SSH key**
4. 粘贴公钥，保存

### 步骤3: 修改远程URL并推送

```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment

# 修改为SSH URL
git remote set-url origin git@github.com:NQ-lovecpp/paddle_detection_mouse.git

# 推送
git push -u origin master
```

---

## 🔐 方式3: 在远程URL中包含Token

**注意**: 这种方式会在配置文件中明文保存token，不太安全。

```bash
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment

# 修改远程URL（替换YOUR_TOKEN）
git remote set-url origin https://YOUR_TOKEN@github.com/NQ-lovecpp/paddle_detection_mouse.git

# 推送
git push -u origin master
```

---

## 📋 方式4: 手动推送（最简单）

如果服务器推送困难，可以在本地Mac/Windows操作：

### 在Mac上操作

```bash
# 1. 克隆远程仓库
cd ~/Projects
git clone https://github.com/NQ-lovecpp/paddle_detection_mouse.git

# 2. 从服务器复制文件到Mac
# (使用rsync或scp，参考之前的传输命令)
rsync -avz user@server:/hy-tmp/paddle_detection_mouse/Mobile_Deployment/ \
  ~/Projects/paddle_detection_mouse/Mobile_Deployment/

# 3. 提交并推送
cd ~/Projects/paddle_detection_mouse
git add Mobile_Deployment/
git commit -m "feat: 添加移动端部署文件和ONNX模型"
git push origin master
```

---

## ⚠️ 常见问题

### Q1: 推送时提示文件太大

```
remote: error: File models/yolov3_mouse_fp32.onnx is 92.34 MB
```

**当前文件**: 92.34 MB  
**GitHub限制**: 100 MB  
**状态**: ✅ 在限制内，应该可以推送

如果仍然失败，使用Git LFS:

```bash
# 安装Git LFS
apt-get install git-lfs

# 配置LFS
cd /hy-tmp/paddle_detection_mouse/Mobile_Deployment
git lfs install
git lfs track "*.onnx"
git add .gitattributes
git commit -m "chore: 配置Git LFS"

# 迁移现有文件到LFS
git lfs migrate import --include="*.onnx" --everything

# 推送
git push -u origin master --force
```

### Q2: 推送被拒绝

```
! [rejected]        master -> master (fetch first)
```

**原因**: 远程仓库有本地没有的提交

**解决**:
```bash
# 拉取远程更改
git pull origin master --rebase

# 推送
git push -u origin master
```

### Q3: 身份验证失败

```
fatal: Authentication failed
```

**解决**:
- 检查token是否正确
- 检查token权限是否包含repo
- 检查token是否过期

---

## ✅ 推送成功验证

推送成功后，检查：

1. **访问仓库**: https://github.com/NQ-lovecpp/paddle_detection_mouse
2. **确认文件**:
   - [ ] Mobile_Deployment/ 目录存在
   - [ ] models/yolov3_mouse_fp32.onnx 已上传
   - [ ] 所有文档文件已上传
   - [ ] 提交历史完整

3. **查看文件大小**:
   - 在GitHub上查看 models/yolov3_mouse_fp32.onnx
   - 应该显示 92.3 MB

---

## 🎯 推荐操作流程

**我推荐使用方式1（Personal Access Token）**:

1. ✅ 创建GitHub Token (5分钟)
2. ✅ 执行推送命令
3. ✅ 验证文件已上传

**如果方式1失败，使用方式4（手动推送）**:
- 在Mac上克隆仓库
- 复制文件
- 本地推送

---

## 📞 需要帮助？

如果你已经有了GitHub Token，告诉我，我可以帮你执行推送命令。

或者你可以按照上面的步骤自己操作。

---

**最后更新**: 2026-02-09

