# 目录

- [TASK 1: Customer Churn Prediction Model](#task-1-customer-churn-prediction-model)
- [TASK 2: Anomaly Detection System](#task-2-anomaly-detection-system)
- [TASK 3: RAG Chatbot](#task-3-rag-chatbot)

---

# TASK 1: Customer Churn Prediction Model

## 1. 数据概览

###  客户总体流失情况

![Churn 总体分布](image/churned.png)

- 流失率约为 30%
- 数据集共1000条样本

### 特征基本分布情况

#### 性别
![Boxplot: Gender vs Churn](image/gender.png)  
#### 余额水平
![Boxplot: Balance vs Churn](image/balance_level.png)
#### 收入
![Boxplot: Income vs Churn](image/income.png)  
#### 满意度得分
![Boxplot: satisfaction_score vs Churn](image/satisfaction_score.png)
#### 投诉记录
![Bar: Complaints vs Churn](image/complaint.png)
#### 支付延迟
![Bar: payment_delay vs Churn](image/payment_delay.png)

### PCA 客户分布

![PCA](image/pca_visualization.png) 

- 客户流失在降维空间中未形成明显聚类，说明流失行为并非由单一线性因素主导；
- 主成分累计解释率仅为 27%，说明原始数据结构较复杂

## 2. 特征工程

**清洗处理**：
- 缺失值/重复值处理：原始数据中无该情况
- 异常值处理：对`tenure_month`做clip，限制其最大值为100个月
- 分类变量编码：One-Hot编码
- 数值变量变换：对 `income` 做对数转换
- 分箱处理：对`account_balance`,`payment_delays_6m`,`service_calls_6m`做分箱

**新增特征：**

| 特征名 | 说明 |
|--------|------|
| `balance_income_ratio` | 余额占收入比例 |
| `balance_satisfaction` | 余额 × 满意度|
| `monthly_spend` | 月交易次数 × 均值|

## 3.模型训练
为确保模型具备稳定性和泛化能力，在训练过程中采用了以下策略：
- 使用 GridSearchCV对每个模型进行系统性超参数调优；

- 所有模型训练均使用 StratifiedKFold 分层抽样的 10 折交叉验证，确保在不同样本划分下性能稳定；

- 对于样本不平衡的问题，部分模型（如 Random Forest、Logistic Regression）通过设置 class_weight，以及对正负样本权重进行调整（如 XGBoost、LightGBM 中的 scale_pos_weight）以提高召回能力；

## 4. 模型比较与评估结果

以下为各模型在测试集下的最佳阈值下评估指标对比：

（以 F1 Score 与 ROC-AUC 为主要综合指标）

| Model               | Best Threshold | Precision | Recall | F1 Score | ROC-AUC |
|--------------------|----------------|-----------|--------|----------|---------|
| Random Forest       | 0.315          | 0.403     | 0.878  | 0.552    | 0.706   |
| Logistic Regression | 0.183          | 0.389     | **0.911**  | 0.545    | 0.696   |
| XGBoost         | 0.497          | **0.481** | 0.722  | **0.578**| **0.722**|
| LightGBM            | 0.453          | 0.447     | 0.789  | 0.570    | 0.710   |



- XGBoost 综合性能最优，F1 Score 与 ROC-AUC 均领先，适合作为生产模型；
- Logistic Regression 的recall 表现最好，适合用于前置风险筛查；
- Random Forest 在 Recall 上也很强（0.878），但 Precision 较低，可能带来更多误报；
- 所有模型的ROC-AUC在0.69-0.73之间，对流失客户识别稳定，可辅助后续营销运营。


## 5. 特征重要性分析

### XGBoost 特征重要性图

![XGBoost Feature Importance](image/xgboost_importance.png)

### Logistic Regression 系数方向图

![Logistic Coefficients](image/lrimportance.png)

| 关键变量 | 影响方向 | 解读 |
|----------|----------|------|
| `satisfaction_score` | 越低越易流失 | 顾客体验至关重要 |
| `balance_satisfaction` | 越低越易流失 | 低余额 + 不满意 是典型风险群体 |
| `complaints_6m` | 有投诉显著提升流失概率 | 服务质量直接影响忠诚度 |
| `payment_delay_level` | Repeated Delay 表现为高风险 | 支付不规律是预警信号 |

---
# TASK 2: Anomaly Detection System


## 技术实现思路

### 1. 数据处理与特征工程

- 时间同步、异常值修正、归一化
- 滑动窗口处理，构造时序结构
- Rolling 统计特征：均值、方差、变化率等
- 对深度模型可直接输入原始窗口序列

### 2. 异常检测算法

#### 无监督方法
- **Isolation Forest**（适合高维稀疏异常场景）
- **AutoEncoder**（重构误差高即为异常）
- **LSTM-VAE**（时序 + 潜变量检测，适合捕捉时间依赖性）

#### 半监督方法
- **One-Class SVM**（学习正常样本的边界）
- **AutoEncoder + 利用标注数据微调检测阈值**
- **模型集成策略**（多模型得分融合 + 异常标签调参）

### 3. 异常解释能力

- **Isolation Forest** → 使用 SHAP 值分析特征贡献
- **AutoEncoder** → 输出每个特征的重构误差
- **LSTM-VAE + Attention** → 注意力权重高的时间步或特征可用于识别主要异常来源


### 4. 阈值优化与评估

- 使用稀疏标注异常进行 F1-score 优化
- 支持动态阈值（如基于重构误差分布自动调整）
- Precision vs Recall平衡，控制误报率

### 5. Pipeline示意图
![pipeline](image/task2.png)

## 相关工具与平台


| 类型 | 工具 | 说明 |
|------|------|------|
| Python库 | PyOD | 支持几十种检测算法，统一接口，适合快速迭代 |
| 云服务 | AWS Lookout for Equipment | 工业传感器专用，开箱即用，无需建模 |
| 云服务 | Azure Anomaly Detector (Multivariate) | 多变量建模、自动学习相关性，适合部署到 Azure 平台 |

## 参考资料

- [PyOD - Python Outlier Detection](https://pyod.readthedocs.io/en/latest/)

- [AWS Lookout for Equipment](https://aws.amazon.com/lookout-for-equipment/)
- [Azure Anomaly Detector](https://learn.microsoft.com/en-us/azure/cognitive-services/anomaly-detector/)
- [LSTM-VAE Anomaly Detection 论文](https://arxiv.org/abs/1903.02407)
- [SHAP Explainability](https://shap.readthedocs.io/en/latest/)

---

# TASK 3: RAG Chatbot

- **前端界面**：使用 `Streamlit`构建，提供聊天 UI、侧边栏控件（控制查询扩展、重排序、索引管理）
- **后端框架**：LangChain+Python
- **RAG 处理流程**：
  - Query Expand: DeepSeek-v3 API
  - Retrieve: BM25 + 向量相似度
  - Rerank: 使用 Cohere API 进行重排序
  - Generate: 使用 DeepSeek-v3 API 生成回答

- **画面展示**：
```html
<video width="640" height="360" controls>
  <source src="video/ragchatbot.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

## 文档预处理

- **原始格式**：PDF 格式的[技术手册](https://support.industry.siemens.com/cs/cn/zh/view/109767345)和[常见问题文档](https://www.ad.siemens.com.cn/productportal/prods/V90_Document/00_Selection/06_FAQ/SelectionFAQ.htm)
- **转换工具**：使用 [LlamaParse](https://cloud.llamaindex.ai/project/9b763bc0-8014-43f7-9128-0f5be4d64ddf/pipeline) 将 PDF 转为结构化 Markdown 文件
- **切块策略**：基于 Markdown 层级结构划分，使用 LangChain的`MarkdownHeaderTextSplitter`：
  - `#` 为一级标题
  - `##` 为二级标题
  - 过滤掉仅包含标题或无实际内容的块


## 向量索引与嵌入

- **嵌入模型**：`BAAI/bge-small-zh-v1.5`，支持中文，运行于 CPU
- **向量索引**：使用 `FAISS`本地构建和保存
- **稀疏索引**：基于 `rank_bm25.BM25Okapi` 实现 BM25 检索器
- **混合索引策略**：利用 LangChain 的 `EnsembleRetriever` 加权组合 BM25 与 FAISS 向量检索


## 检索策略

- **查询扩展**：启用后，使用 LLM 从用户提问中识别关键术语并生成相关术语组合查询，提升召回率
- **文档检索流程**：
  - 使用 BM25 与 FAISS 混合检索相关文档块
  - 可选启用 Cohere Rerank API 对结果进行重排序
  - 按文档相关性得分生成置信度评级（高 / 中 / 低）


## 回答生成

- **语言模型**：使用 DeepSeek-v3 API
- **系统提示词模板**：引导 LLM 生成专业、准确、引用充分的回答
- **上下文注入方式**：将 Top N 文档块（含引用和置信度）拼接后注入 prompt 中，辅助生成

---



