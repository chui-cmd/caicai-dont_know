# caicai-dont_know

数学建模代码仓库，用于存储建模比赛相关的 Python 代码、实验脚本、数据处理流程、结果输出和报告材料。

## 目录结构

- `src/`: 通用 Python 代码和模型实现
- `scripts/`: 可直接运行的实验脚本
- `scripts/数据预处理/`: 数据读取、清洗、标准化和相关性分析
- `scripts/二/`: 综合评价与权重计算方法
- `scripts/三/`: 后续第三类模型与分析脚本
- `scripts/四/`: 后续第四类模型与分析脚本
- `scripts/五/`: 后续第五类模型与分析脚本
- `src/科研绘图配色.py`: 统一科研图表配色与绘图风格
- `notebooks/`: Jupyter Notebook 分析过程
- `data/`: 本地数据文件，默认不提交到 GitHub
- `outputs/`: 图表、预测结果等输出文件，默认不提交到 GitHub
- `docs/`: 建模思路、公式推导和报告材料

## 已有脚本

- `九九乘法表.py`: 九九乘法表 Python 示例
- `scripts/数据预处理/指标预处理模板.py`: 指标正向化、Z-score 标准化、极差标准化模板
- `scripts/数据预处理/相关性热力图.py`: Pearson 与 Spearman 相关性热力图模板
- `scripts/数据预处理/数据读取与缺失异常值填补.py`: 数据读取、缺失值填补、异常值检测与处理模板
- `scripts/二/层次分析法全过程.py`: AHP 层次分析法权重计算、一致性检验与方案评分模板
- `scripts/二/熵权法全过程.py`: 熵权法正向化、归一化、权重计算与综合评分模板
- `scripts/二/TOPSIS排序全过程.py`: TOPSIS 正负理想解、贴近度计算与方案排序模板
- `scripts/二/熵权法加TOPSIS全过程.py`: 熵权法自动赋权与 TOPSIS 综合排序模板
- `scripts/三/多元线性回归全过程.py`: 多元线性回归系数、R2、p值与残差诊断图模板
- `scripts/三/GM11灰色预测全过程.py`: GM(1,1) 灰色预测、级比检验、误差评价与预测图模板
- `scripts/三/插值与最小二乘拟合全过程.py`: 一维插值、多项式最小二乘、非线性最小二乘拟合模板
- `scripts/四/线性规划全过程.py`: 线性规划建模、求解、目标值与约束松弛量分析模板
- `scripts/四/整数规划与01规划全过程.py`: 整数规划与0-1规划建模、求解和结果分析模板
- `scripts/五/Kmeans聚类全过程.py`: K-means聚类、K值选择、PCA可视化与聚类画像模板
- `scripts/五/分类流程全过程.py`: 分类建模、训练测试划分、评价指标、混淆矩阵与特征重要性模板

## 统一图表配色

所有绘图脚本统一使用科研配色方案：

- 红色：`#C02635`
- 珊瑚色：`#F58966`
- 深蓝色：`#00609D`
- 浅蓝色：`#7EABCE`
- 橙色：`#F5A86E`
- 浅黄色：`#F4D291`

## 常用命令

```powershell
pip install -r requirements.txt
python 九九乘法表.py
python scripts/数据预处理/指标预处理模板.py
python scripts/数据预处理/相关性热力图.py
python scripts/数据预处理/数据读取与缺失异常值填补.py
python scripts/二/层次分析法全过程.py
python scripts/二/熵权法全过程.py
python scripts/二/TOPSIS排序全过程.py
python scripts/二/熵权法加TOPSIS全过程.py
python scripts/三/多元线性回归全过程.py
python scripts/三/GM11灰色预测全过程.py
python scripts/三/插值与最小二乘拟合全过程.py
python scripts/四/线性规划全过程.py
python scripts/四/整数规划与01规划全过程.py
python scripts/五/Kmeans聚类全过程.py
python scripts/五/分类流程全过程.py
git status
git add .
git commit -m "Update modeling code"
git push
```
