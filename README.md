# 目录

- [TASK 1: Customer Churn Prediction Model](#task-1customer-churn-prediction-model)
- [TASK 2: Anomaly Detection System](#task2-anomaly-detection-system)

---

# TASK 1:Customer Churn Prediction Model

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
| Random Forest       | 0.344          | 0.415     | 0.817  | 0.551    | 0.702   |
| Logistic Regression | 0.202          | 0.378     | 0.850  | 0.523    | 0.676   |
| XGBoost             | 0.502          | 0.434     | 0.767  | **0.554**| **0.709**|
| LightGBM            | 0.484          | 0.412     | 0.783  | 0.540    | 0.696   |



- XGBoost 综合性能最优，F1 Score 与 ROC-AUC 均领先，适合作为生产模型；
- Logistic Regression 的recall 表现最好，适合用于前置风险筛查；
- 所有模型的ROC-AUC在0.67-0.71之间，对流失客户识别稳定，可辅助后续营销运营。


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
# Task2:Anomaly Detection System


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
