# 前端重构说明

## 概述

前端已经完全重构，移除了股票和Polymarket相关功能，现在专注于展示**研究想法排行榜**。

## 主要变更

### 1. 移除的功能
- ❌ 股票交易仪表板 (`StockDashboard.tsx`)
- ❌ Polymarket仪表板 (`PolymarketDashboard.tsx`)
- ❌ 新闻页面 (`News.tsx`)
- ❌ 社交媒体页面 (`SocialMedia.tsx`)
- ❌ 投资组合展示 (`Portfolio.tsx`)
- ❌ 系统监控 (`SystemMonitoring.tsx`)
- ❌ 模型展示 (`ModelsDisplay.tsx`)

### 2. 新增功能
- ✅ **研究想法排行榜** - 展示按影响力排序的研究想法
- ✅ 简化的导航栏（仅包含"研究想法"和"关于"）
- ✅ 现代化的卡片式设计
- ✅ 响应式布局，支持移动端和桌面端
- ✅ 实时数据更新（每5分钟）

### 3. 保留的功能
- ✅ 关于页面 (`About.tsx`)
- ✅ 页脚 (`Footnote.tsx`)
- ✅ 导航栏 (`Navigation.tsx` - 已简化)
- ✅ 浏览量统计

## 文件结构

```
frontend/
├── src/
│   ├── App.tsx                    # 主应用入口（已简化）
│   ├── App.css                    # 全局样式
│   ├── types.ts                   # TypeScript类型定义（已简化）
│   ├── components/
│   │   ├── Dashboard.tsx          # 研究想法排行榜（重构）
│   │   ├── Dashboard.css          # 排行榜样式（重构）
│   │   ├── Navigation.tsx         # 导航栏（已简化）
│   │   ├── About.tsx             # 关于页面
│   │   ├── About.css
│   │   ├── Footnote.tsx          # 页脚
│   │   └── Footnote.css
│   └── ...
└── ...
```

## 数据结构

### ResearchIdea 类型

```typescript
export interface ResearchIdea {
  id: string;                 // 唯一标识符
  title: string;              // 想法标题
  description: string;        // 想法描述
  author: string;             // 作者姓名
  institution?: string;       // 机构（可选）
  tags: string[];            // 标签列表
  upvotes: number;           // 点赞数
  created_at: string;        // 创建时间
  updated_at?: string;       // 更新时间（可选）
  url?: string;              // 详情链接（可选）
  citations?: number;        // 引用次数（可选）
  impact_score?: number;     // 影响力评分（可选）
}
```

## API 端点

前端期望后端提供以下API端点：

### 1. 获取研究想法列表
```
GET /api/research-ideas
```

**响应示例：**
```json
[
  {
    "id": "1",
    "title": "基于大语言模型的自动化代码审查",
    "description": "利用GPT-4和Claude等大语言模型实现代码质量自动化审查，提高开发效率",
    "author": "张三",
    "institution": "清华大学",
    "tags": ["AI", "代码审查", "自动化"],
    "upvotes": 156,
    "created_at": "2025-10-20T10:00:00Z",
    "citations": 23,
    "impact_score": 8.5,
    "url": "https://example.com/research/1"
  },
  {
    "id": "2",
    "title": "量子计算在药物研发中的应用",
    "description": "探索量子计算如何加速新药研发过程，缩短从实验室到临床的时间",
    "author": "李四",
    "institution": "北京大学",
    "tags": ["量子计算", "生物医药", "AI"],
    "upvotes": 142,
    "created_at": "2025-10-19T15:30:00Z",
    "citations": 18,
    "impact_score": 7.8
  }
]
```

### 2. 浏览量统计（保留）
```
GET /api/views    # 获取浏览量
POST /api/views   # 增加浏览量
```

## 启动方法

### 开发环境

```bash
cd frontend
npm install
npm start
```

应用将在 `http://localhost:3000` 启动。

### 生产构建

```bash
cd frontend
npm run build
```

构建产物将输出到 `frontend/build/` 目录。

## 主要特性

### 1. 排行榜功能
- **排名显示**：前三名有特殊的金银铜徽章
- **评分系统**：支持点赞数、引用次数和影响力评分
- **标签系统**：每个想法可以有多个标签
- **作者信息**：显示作者和所属机构
- **时间戳**：显示相对时间（如"2小时前"）

### 2. 响应式设计
- **桌面端**：宽屏展示，卡片排列紧凑
- **平板端**：中等屏幕优化
- **移动端**：垂直堆叠，触摸友好

### 3. 交互功能
- **点击跳转**：点击卡片可跳转到详情页面（如果有URL）
- **展开/收起**：超过10个想法时显示"查看全部"按钮
- **实时更新**：每5分钟自动刷新数据

## 样式主题

### 颜色方案
- **主色调**：紫色系 (#9c9ef8, #818cf8)
- **背景色**：深色主题 (#0f1419)
- **文字色**：白色和灰色渐变
- **强调色**：
  - 金色 (#ffd700) - 第一名
  - 银色 (#c0c0c0) - 第二名
  - 铜色 (#cd7f32) - 第三名

### 设计元素
- **卡片**：半透明背景，渐变边框
- **悬停效果**：上浮和阴影
- **徽章**：圆角矩形，带阴影
- **标签**：小圆角，半透明背景

## 待办事项

如果需要进一步增强功能，可以考虑：

1. **筛选功能**：按标签、作者或机构筛选
2. **搜索功能**：全文搜索研究想法
3. **排序选项**：按时间、点赞数或影响力排序
4. **详情页面**：点击想法查看完整详情
5. **用户交互**：允许用户点赞或评论
6. **数据可视化**：添加图表展示趋势

## 注意事项

1. **API兼容性**：确保后端API返回正确的数据格式
2. **错误处理**：目前基本的错误处理已实现，可以进一步增强
3. **性能优化**：大量数据时考虑分页加载
4. **SEO优化**：考虑使用SSR或预渲染提高SEO

## 技术栈

- **React** 19.1.0
- **TypeScript** 4.9.5
- **React Router** 7.6.3
- **CSS** (原生，无UI框架)

## 联系方式

如有问题或建议，请查看关于页面或提交Issue。

