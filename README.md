# caicai-dont_know

数学建模代码仓库，用于存储建模比赛相关的 Python 代码、实验脚本、数据处理流程、结果输出和报告材料。

## 目录结构

- `src/`: 通用 Python 代码和模型实现
- `scripts/`: 可直接运行的实验脚本
- `notebooks/`: Jupyter Notebook 分析过程
- `data/`: 本地数据文件，默认不提交到 GitHub
- `outputs/`: 图表、预测结果等输出文件，默认不提交到 GitHub
- `docs/`: 建模思路、公式推导和报告材料

## 已有脚本

- `九九乘法表.py`: 九九乘法表 Python 示例
- `scripts/指标预处理模板.py`: 指标正向化、Z-score 标准化、极差标准化模板
- `scripts/相关性热力图.py`: Pearson 与 Spearman 相关性热力图模板
- `scripts/数据读取与缺失异常值填补.py`: 数据读取、缺失值填补、异常值检测与处理模板

## 常用命令

```powershell
pip install -r requirements.txt
python 九九乘法表.py
python scripts/指标预处理模板.py
python scripts/相关性热力图.py
python scripts/数据读取与缺失异常值填补.py
git status
git add .
git commit -m "Update modeling code"
git push
```
