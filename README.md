# FiberWatch

OTDR（光时域反射仪）光纤事件自动检测与分析系统。

## 安装

```bash
cd FiberWatch
pip install -e .
```

## 项目结构

```
FiberWatch/
├── fiberwatch/
│   ├── config/
│   │   ├── settings.py          # 配置类定义（DetectionConfig 等）
│   │   ├── detection.yaml       # 检测算法参数配置（修改此文件调整检测行为）
│   │   └── llm_config.py        # LLM 分析配置
│   ├── core/
│   │   ├── detector.py          # 检测器主类（Detector）
│   │   ├── models.py            # 数据类（DetectedEvent, DetectionResult）
│   │   └── detection/           # 检测算法子模块
│   │       ├── context.py       #   检测上下文（参数 + 预计算）
│   │       ├── peaks.py         #   反射峰查找
│   │       ├── bend.py          #   弯折检测
│   │       ├── fiber_break.py   #   断纤检测（严重/普通/小峰）
│   │       ├── dirty_connector.py  # 脏污连接器检测
│   │       ├── baseline.py      #   基线拟合
│   │       └── range_finder.py  #   有效检测范围计算
│   ├── scripts/
│   │   ├── visualize.py         # 命令行可视化脚本
│   │   └── visualize_with_llm.py  # 带 LLM 风险预警的可视化
│   ├── utils/
│   │   ├── data_io.py           # 数据读取工具
│   │   ├── visualization.py     # 绘图工具
│   │   └── llm_analyzer.py      # LLM 分析器
│   └── web/
│       ├── app.py               # Streamlit Web 界面
│       └── ui_components.py     # UI 组件
└── pyproject.toml
```

## 使用方式

### 1. 命令行可视化

```bash
# 基本用法：检测 + 生成图表
python -m fiberwatch.scripts.visualize \
    --input_file data/test.txt \
    --baseline data/baseline.txt \
    -o output

# 指定采样间距（默认 0.0025545 km ≈ 2.55 m）
python -m fiberwatch.scripts.visualize \
    --input_file data/test.txt \
    --sample-spacing 0.0025545

# 使用 CNN 数据辅助弯折检测
python -m fiberwatch.scripts.visualize \
    --input_file data/test.txt \
    --baseline data/baseline.txt \
    --cnn-file data/cnn_output.txt

# 不保存图表，仅输出检测结果
python -m fiberwatch.scripts.visualize \
    --input_file data/test.txt \
    --no-save
```

### 2. 带 LLM 风险预警

当算法未检测到事件时，自动调用 LLM 对波形进行风险分析。

```bash
# 需要设置环境变量 DASHSCOPE_API_KEY

# 仅使用算法检测 + LLM 风险预警
python -m fiberwatch.scripts.visualize_with_llm \
    --input_file data/test.txt

# 可选：加入 CNN 数据辅助弯折检测
python -m fiberwatch.scripts.visualize_with_llm \
    --input_file data/test.txt \
    --cnn-file data/cnn_output.txt
```

> CNN 文件（`--cnn-file`）和基线文件（`--baseline`）均为可选参数，不提供时只使用原始算法检测。


### 3. Web 界面

```bash
# 启动 Streamlit Web 界面
python -m fiberwatch.web.app

# 或通过 CLI
fiberwatch web --host localhost --port 8501
```

### 4. Python API 调用

```python
import numpy as np
from fiberwatch.core import Detector
from fiberwatch.utils.data_io import load_test_data

# 加载数据
trace = load_test_data("data/test.txt")

# 创建检测器并运行检测
detector = Detector(
    trace_db=trace,
    sample_spacing_km=0.0025545,  # 采样间距
)
result = detector.detect(trace)

# 查看检测结果
for event in result.events:
    print(f"{event.kind} @ {event.z_km:.4f} km, "
          f"loss={event.magnitude_db:.2f} dB")

# 查看反射峰
for peak in result.reflection_peaks:
    print(f"Reflection @ index {peak['index']}, "
          f"height={peak['peak_height_db']:.1f} dB")

# 生成可视化图表
result.plot(outfile="output/result.png")
```

#### 使用基线数据

```python
baseline = load_test_data("data/baseline.txt")

detector = Detector(
    trace_db=trace,
    baseline=baseline,
    sample_spacing_km=0.0025545,
)
result = detector.detect(trace)
```

#### 自定义检测参数

```python
from fiberwatch.config import DetectionConfig

# ：从 YAML 文件加载
config = DetectionConfig.from_yaml("my_config.yaml")

## 检测参数配置

所有检测参数集中在 `fiberwatch/config/detection.yaml` 中，修改此文件即可调整检测行为，无需改动代码。

主要参数分类：

| 分类 | 说明 |
|------|------|
| 反射峰相关 | 峰查找的宽度、高度、突出度阈值 |
| 阶梯下降 | 弯折/断纤判定的下降幅度阈值 |
| 噪声判断 | 噪声底值、标准差阈值 |
| 断纤参数 | 严重/普通/小峰断纤的判定条件 |
| 范围控制 | 有效检测区间的起止位置 |

## 检测事件类型

| 事件类型 | kind 值 | 说明 |
|---------|---------|------|
| 严重断纤 | `severe_break` | 信号大幅下降并进入噪声 |
| 普通断纤 | `break` | 阶梯式信号下降 |
| 小峰断纤 | `small_peak_break` | 小反射峰伴随信号下降 |
| 弯折 | `bend` | 光纤弯曲导致的损耗 |
| 脏污连接器 | `dirty_connector` | 连接器污染导致的反射 |

## 输入数据格式

OTDR 数据文件为纯文本格式，每行一个采样值（dB），例如：

```
-10.234
-10.235
-10.238
...
```
