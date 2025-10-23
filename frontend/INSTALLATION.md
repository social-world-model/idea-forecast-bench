# 安装指南

## 📋 前置要求

- **Node.js**: >= 16.0.0 (推荐 LTS 版本)
- **npm**: >= 7.0.0 (通常随 Node.js 安装)
- **操作系统**: macOS, Linux, Windows

检查版本：

```bash
node --version
npm --version
```

## 🔧 安装步骤

### 1. 导航到前端目录

```bash
cd /Users/yuhaofei/Downloads/live-idea-bench/frontend
```

### 2. 清理旧的依赖（如果存在）

```bash
rm -rf node_modules package-lock.json
```

### 3. 安装依赖

```bash
npm install
```

这可能需要几分钟时间，请耐心等待。

### 4. 验证安装

```bash
# 使用模拟数据启动（推荐）
REACT_APP_USE_MOCK_DATA=true npm start
```

如果一切正常，浏览器会自动打开 http://localhost:3000。

## 🚀 启动选项

### 选项 A: 使用模拟数据（无需后端）

```bash
# Mac/Linux
REACT_APP_USE_MOCK_DATA=true npm start

# Windows (PowerShell)
$env:REACT_APP_USE_MOCK_DATA="true"; npm start

# Windows (CMD)
set REACT_APP_USE_MOCK_DATA=true && npm start
```

### 选项 B: 连接真实后端

确保后端服务运行在 http://localhost:5001，然后：

```bash
npm start
```

### 选项 C: 使用环境变量文件

创建 `.env.local` 文件：

```bash
echo "REACT_APP_USE_MOCK_DATA=true" > .env.local
npm start
```

## 🏗️ 构建生产版本

```bash
npm run build
```

构建产物将在 `build/` 目录中。

## 🐛 常见问题

### Q1: npm install 失败

**错误示例**：
```
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
```

**解决方案**：
```bash
# 清理缓存
npm cache clean --force

# 删除 node_modules
rm -rf node_modules package-lock.json

# 使用 legacy peer deps 安装
npm install --legacy-peer-deps
```

### Q2: 端口 3000 被占用

**错误示例**：
```
Something is already running on port 3000
```

**解决方案**：
```bash
# 方案1: 使用其他端口
PORT=3001 npm start

# 方案2: 杀死占用端口的进程
# Mac/Linux
lsof -ti:3000 | xargs kill -9

# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

### Q3: 模块找不到错误

**错误示例**：
```
Error: Cannot find module 'react-scripts'
```

**解决方案**：
```bash
# 重新安装
npm install

# 如果还不行，尝试
npm install react-scripts --save-dev
```

### Q4: TypeScript 编译错误

**解决方案**：
```bash
# 确保 TypeScript 已安装
npm install typescript --save-dev

# 重启开发服务器
npm start
```

### Q5: 白屏问题

**可能原因**：
1. JavaScript 错误
2. API 连接失败（非模拟数据模式）
3. 浏览器不兼容

**解决方案**：
```bash
# 1. 打开浏览器控制台查看错误
# 2. 尝试使用模拟数据模式
REACT_APP_USE_MOCK_DATA=true npm start

# 3. 清理浏览器缓存
# 4. 尝试使用 Chrome 浏览器
```

## 📱 浏览器要求

### 推荐浏览器
- Chrome (最新版) ✅
- Firefox (最新版) ✅
- Safari (最新版) ✅
- Edge (最新版) ✅

### 最低要求
- Chrome >= 90
- Firefox >= 88
- Safari >= 14
- Edge >= 90

## 🔍 验证安装

### 检查清单

- [ ] Node.js 版本 >= 16
- [ ] npm 版本 >= 7
- [ ] `npm install` 成功完成
- [ ] `npm start` 启动成功
- [ ] 浏览器能打开 http://localhost:3000
- [ ] 页面显示研究想法排行榜
- [ ] 控制台无严重错误

### 测试命令

```bash
# 检查 TypeScript 编译
npx tsc --noEmit

# 检查代码格式（如果配置了）
npm run lint

# 运行测试
npm test
```

## 📊 依赖说明

### 主要依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| react | 19.1.0 | UI 框架 |
| react-dom | 19.1.0 | React DOM 渲染 |
| react-router-dom | 7.6.3 | 路由管理 |
| typescript | 4.9.5 | 类型系统 |
| react-scripts | 5.0.1 | 构建工具 |

### 开发依赖

所有开发依赖已在 `package.json` 中配置。

## 🔄 更新依赖

```bash
# 检查可更新的包
npm outdated

# 更新所有包到最新版本（谨慎操作）
npm update

# 更新特定包
npm update react react-dom
```

## 🗑️ 清理

### 完全清理

```bash
# 删除所有生成的文件
rm -rf node_modules
rm -rf build
rm package-lock.json
rm -rf .env.local
```

### 重新开始

```bash
# 清理后重新安装
npm install
REACT_APP_USE_MOCK_DATA=true npm start
```

## 💡 开发提示

### 热重载

代码修改后会自动重载，无需手动刷新浏览器。

### 调试

1. 安装 React DevTools 浏览器扩展
2. 打开浏览器开发者工具 (F12)
3. 查看 Console 标签页的日志

### 性能

首次编译可能较慢（1-2分钟），后续修改编译会很快。

## 🆘 获取帮助

如果以上方案都无法解决问题：

1. 检查 Node.js 和 npm 版本是否满足要求
2. 查看完整的错误信息
3. 搜索错误信息寻找解决方案
4. 提交 Issue 并附上：
   - 操作系统版本
   - Node.js 版本
   - npm 版本
   - 完整错误信息
   - 执行的命令

## 📚 相关文档

- [快速启动指南](./QUICK_START.md)
- [重构说明](./REBUILD_README.md)
- [主 README](./README.md)

---

**提示**: 首次安装建议使用模拟数据模式，可以快速验证安装是否成功！

