# 研究想法排行榜 - 前端

这是一个基于 React + TypeScript 构建的研究想法排行榜展示平台。

## 🚀 快速开始

### 使用模拟数据（推荐用于开发）

无需后端API，使用预设的模拟数据：

```bash
npm install
REACT_APP_USE_MOCK_DATA=true npm start
```

### 连接真实API

```bash
npm install
npm start
```

应用将在 [http://localhost:3000](http://localhost:3000) 启动。

> 💡 **提示**: 首次启动建议使用模拟数据模式，快速查看效果。

## 📚 文档

- **[快速启动指南](./QUICK_START.md)** - 详细的启动和配置说明
- **[重构说明](./REBUILD_README.md)** - 完整的架构和API文档
- **[变更总结](./CHANGES_SUMMARY.md)** - 重构变更详细列表

## ✨ 主要特性

- 🏆 **排行榜展示** - 按影响力排序的研究想法
- 🎨 **现代化UI** - 卡片式设计，渐变和阴影效果
- 📱 **响应式** - 支持桌面、平板和移动设备
- 🔄 **实时更新** - 可配置的自动刷新
- 🧪 **模拟数据** - 内置测试数据，无需后端即可开发
- ⚡ **高性能** - 优化的加载和渲染性能

## 🛠️ 可用命令

### `npm start`

启动开发服务器，支持热重载。

### `npm run build`

构建生产版本到 `build/` 目录。

### `npm test`

运行测试套件（交互式监视模式）。

## 📦 技术栈

- **React** 19.1.0 - UI框架
- **TypeScript** 4.9.5 - 类型安全
- **React Router** 7.6.3 - 路由管理
- **Create React App** 5.0.1 - 构建工具

## 🎯 项目结构

```
frontend/
├── src/
│   ├── App.tsx                 # 主应用组件
│   ├── App.css                 # 全局样式
│   ├── types.ts                # TypeScript类型定义
│   ├── config.ts               # 配置管理
│   ├── mockData.ts             # 模拟数据
│   ├── components/
│   │   ├── Dashboard.tsx       # 排行榜组件
│   │   ├── Dashboard.css       # 排行榜样式
│   │   ├── Navigation.tsx      # 导航栏
│   │   ├── About.tsx           # 关于页面
│   │   └── Footnote.tsx        # 页脚
│   └── ...
├── public/                     # 静态资源
├── package.json                # 依赖配置
└── tsconfig.json              # TypeScript配置
```

## ⚙️ 环境变量

创建 `.env.local` 文件进行自定义配置：

```bash
# 使用模拟数据（开发推荐）
REACT_APP_USE_MOCK_DATA=true

# 自定义API地址
REACT_APP_API_BASE_URL=http://localhost:5001

# 刷新间隔（毫秒，默认5分钟）
REACT_APP_REFRESH_INTERVAL=300000
```

## 🔌 后端API要求

如果连接真实API，后端需要提供以下端点：

### 获取研究想法列表

```
GET /api/research-ideas
```

**响应格式**:
```json
[
  {
    "id": "1",
    "title": "研究想法标题",
    "description": "详细描述",
    "author": "作者名",
    "institution": "机构名",
    "tags": ["标签1", "标签2"],
    "upvotes": 100,
    "created_at": "2025-10-23T10:00:00Z",
    "citations": 10,
    "impact_score": 8.5,
    "url": "https://example.com"
  }
]
```

### 浏览量统计

```
GET /api/views     # 获取浏览量
POST /api/views    # 增加浏览量
```

详细API文档请参考 [REBUILD_README.md](./REBUILD_README.md)。

## 🎨 自定义样式

主要样式文件：

- `src/App.css` - 全局样式、导航栏
- `src/components/Dashboard.css` - 排行榜、卡片样式
- `src/index.css` - 基础样式重置

## 🧪 开发提示

### 使用模拟数据

开发新功能时使用模拟数据可以避免依赖后端：

```bash
# 临时使用
REACT_APP_USE_MOCK_DATA=true npm start

# 或修改 .env.local
echo "REACT_APP_USE_MOCK_DATA=true" > .env.local
npm start
```

### 修改模拟数据

编辑 `src/mockData.ts` 来自定义测试数据。

### 查看调试日志

开发模式下会在控制台输出详细日志：

```
[App] 🔄 Fetching research ideas...
[App] 📦 Using mock data
[App] ✅ Research ideas updated: 15 ideas
```

## 🐛 故障排除

### 端口被占用

如果3000端口被占用，可以指定其他端口：

```bash
PORT=3001 npm start
```

### 无法连接后端

1. 确保后端服务正在运行
2. 检查 `package.json` 中的 `proxy` 配置
3. 或使用模拟数据模式：`REACT_APP_USE_MOCK_DATA=true npm start`

### 构建失败

```bash
# 清理并重新安装
rm -rf node_modules package-lock.json
npm install
npm start
```

## 📖 学习资源

- [React 官方文档](https://react.dev/)
- [TypeScript 手册](https://www.typescriptlang.org/docs/)
- [Create React App 文档](https://create-react-app.dev/)
- [React Router 文档](https://reactrouter.com/)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

参见项目根目录的 LICENSE 文件。

---

**版本**: 2.0.0  
**最后更新**: 2025-10-23
