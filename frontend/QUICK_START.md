# 快速启动指南

## 🚀 快速开始

### 方式一：使用模拟数据（无需后端）

这是最快的启动方式，适合前端开发和测试。

```bash
cd frontend
npm install
REACT_APP_USE_MOCK_DATA=true npm start
```

应用将在 `http://localhost:3000` 启动，使用预设的模拟数据。

### 方式二：连接真实后端API

```bash
cd frontend
npm install
npm start
```

默认会尝试连接 `http://localhost:5001` 的后端API。

## 🔧 环境变量配置

创建 `.env.local` 文件来自定义配置：

```bash
# 使用模拟数据
REACT_APP_USE_MOCK_DATA=true

# 自定义API地址
REACT_APP_API_BASE_URL=http://your-backend-server:port

# 自定义刷新间隔（毫秒）
REACT_APP_REFRESH_INTERVAL=300000
```

## 📦 构建生产版本

```bash
cd frontend
npm run build
```

构建产物将输出到 `build/` 目录。

## 🧪 测试模式

### 临时使用模拟数据

```bash
REACT_APP_USE_MOCK_DATA=true npm start
```

### 永久使用模拟数据

创建 `.env.local` 文件：

```bash
echo "REACT_APP_USE_MOCK_DATA=true" > .env.local
npm start
```

## 📝 修改模拟数据

编辑 `src/mockData.ts` 文件来自定义模拟数据：

```typescript
export const mockResearchIdeas: ResearchIdea[] = [
  {
    id: '1',
    title: '你的研究想法标题',
    description: '详细描述...',
    author: '作者名',
    institution: '机构名',
    tags: ['标签1', '标签2'],
    upvotes: 100,
    created_at: '2025-10-23T10:00:00Z',
    citations: 10,
    impact_score: 8.5,
    url: 'https://example.com'
  },
  // 添加更多...
];
```

## 🎨 自定义样式

主要样式文件：

- `src/App.css` - 全局样式和导航栏
- `src/components/Dashboard.css` - 排行榜样式
- `src/index.css` - 基础样式

## 🔍 调试

开发模式下，控制台会显示详细日志：

```
[App] 🔄 Fetching research ideas...
[App] 📦 Using mock data
[App] ✅ Research ideas updated: 15 ideas
```

生产模式下，只显示错误日志。

## 📱 支持的浏览器

- Chrome (推荐)
- Firefox
- Safari
- Edge

## ⚡ 性能优化建议

1. **懒加载**：大型组件可以使用 React.lazy
2. **分页**：数据量大时建议实现分页
3. **缓存**：考虑使用 Service Worker 缓存静态资源
4. **CDN**：生产环境建议使用CDN加速

## 🐛 常见问题

### Q: 启动后显示空白页面？

A: 检查控制台错误信息，确保：
1. Node.js 版本 >= 16
2. 依赖已正确安装 (`npm install`)
3. 端口3000未被占用

### Q: 无法连接到后端API？

A: 有两个选择：
1. 使用模拟数据模式：`REACT_APP_USE_MOCK_DATA=true npm start`
2. 确保后端服务正在运行，并检查 `package.json` 中的 `proxy` 设置

### Q: 如何修改刷新间隔？

A: 在 `.env.local` 中设置：
```bash
REACT_APP_REFRESH_INTERVAL=180000  # 3分钟
```

### Q: 如何部署到生产环境？

A: 
```bash
npm run build
# 将 build/ 目录部署到静态服务器
# 或使用 serve 工具测试：
npx serve -s build
```

## 📚 更多文档

- [完整重构说明](./REBUILD_README.md)
- [React 文档](https://react.dev/)
- [TypeScript 文档](https://www.typescriptlang.org/)

## 💡 提示

- 使用 React DevTools 浏览器扩展来调试组件
- 开发时打开浏览器的控制台查看日志
- 修改代码后会自动热重载，无需手动刷新

