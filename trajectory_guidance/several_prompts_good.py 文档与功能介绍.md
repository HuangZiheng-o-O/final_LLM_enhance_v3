# several_prompts_good.py 文档与功能介绍

## 概述

该 Python 脚本旨在处理轨迹描述，使用 OpenAI 的 GPT-4 模型生成并执行用于曲线生成的 Python 代码，插值并重新采样生成的点，计算速度向量，并保存处理后的数据以供后续使用。脚本模块化，包含处理特定任务的函数，如占位符替换、代码提取、曲线插值以及交互日志记录。

### 主要功能：

1. **占位符替换**：动态替换指令模板中的占位符为实际的轨迹描述。
2. **代码提取与执行**：从 OpenAI 的响应中提取 Python 代码块并执行，以获取曲线定义。
3. **曲线插值与重新采样**：使用 B 样条插值平滑并重新采样曲线为均匀分布的点。
4. **速度计算**：基于重新采样的点计算速度向量。
5. **数据日志记录与保存**：记录交互日志并将处理后的速度数据保存为 `.npy` 格式以供后续使用。

---

## 目录

1. [依赖项](#依赖项)
2. [配置](#配置)
3. [函数](#函数)
    - [replace_placeholders](#replace_placeholders)
    - [format_keyframes](#format_keyframes)
    - [replace_pattern_name](#replace_pattern_name)
    - [extract_code_block](#extract_code_block)
    - [interpolate_and_resample](#interpolate_and_resample)
    - [execute_code_and_compute_keyframes](#execute_code_and_compute_keyframes)
    - [interpolate_and_resample_velocity](#interpolate_and_resample_velocity)
    - [interpolate_and_resample_velocity2](#interpolate_and_resample_velocity2)
    - [inside](#inside)
    - [log_results](#log_results)
    - [main](#main)
4. [执行流程](#执行流程)
5. [错误处理](#错误处理)
6. [使用示例](#使用示例)

---

## 依赖项

脚本依赖以下 Python 库：

- **标准库**：
    - `os`
    - `math`
    - `re`
    - `typing`
- **第三方库**：
    - `yaml`：用于解析 YAML 配置文件。
    - `numpy`：用于数值计算。
    - `matplotlib.pyplot`：用于绘图（在脚本中已导入但未使用）。
    - `scipy`：用于插值、积分和优化。
    - `openai`：用于与 OpenAI 的 GPT-4 模型交互。

**安装**：
确保所有依赖项已安装。可以使用 `pip` 进行安装：

```bash
pip install pyyaml numpy scipy openai matplotlib
```

---

## 配置

脚本期望在同一目录下存在一个 `config.yaml` 文件，其中包含 OpenAI 的 API 密钥。`config.yaml` 的结构如下：

```yaml
token: YOUR_OPENAI_API_KEY
```

将 `YOUR_OPENAI_API_KEY` 替换为您的实际 OpenAI API 密钥。

---

## 函数

### 1. `replace_placeholders`

**用途**：将指令内容中的占位符替换为实际的描述。

**签名**：
```python
def replace_placeholders(instruct_content: str, descriptions: List[str]) -> str:
```

**参数**：
- `instruct_content` (`str`)：包含占位符（如 `placeholder1`、`placeholder2` 等）的指令模板。
- `descriptions` (`List[str]`)：用于替换占位符的描述列表。

**返回值**：
- `str`：占位符已被相应描述替换后的指令内容。

**示例**：
```python
instruct_content = "基于 placeholder1 和 placeholder2 创建一个模式。"
descriptions = ["圆形轨迹", "螺旋运动"]
result = replace_placeholders(instruct_content, descriptions)
# 结果: "基于圆形轨迹和螺旋运动创建一个模式。"
```

---

### 2. `format_keyframes`

**用途**：将关键帧列表格式化为适合代码插入的结构化字符串。

**签名**：
```python
def format_keyframes(keyframes: list, pattern_name="origin_30points", keyframes_str="keyframes_str") -> str:
```

**参数**：
- `keyframes` (`list`)：关键帧列表，每个关键帧为 `(frame_number, (x_coordinate, y_coordinate))` 形式的元组。
- `pattern_name` (`str`, 可选)：模式的基本名称。默认为 `"origin_30points"`。
- `keyframes_str` (`str`, 可选)：要附加到模式名称的额外字符串。默认为 `"keyframes_str"`。

**返回值**：
- `str`：表示关键帧的格式化字符串，适用于代码中使用。

**示例**：
```python
keyframes = [
    (0, (0.00, 0.00)),
    (1, (1.23, 4.56)),
    # ... 更多关键帧
]
formatted = format_keyframes(keyframes)
# 结果:
# elif pattern == "origin_30points_keyframes_str":
#     kframes = [
#         (0, (0.00, 0.00)),
#         (1, (1.23, 4.56)),
#         ...
#     ]
```

---

### 3. `replace_pattern_name`

**用途**：将后续问题中的特定占位符替换为实际的模式名称。

**签名**：
```python
def replace_pattern_name(follow_up_question: str, pattern_name: str) -> str:
```

**参数**：
- `follow_up_question` (`str`)：包含占位符 `PALCEHOLDER2` 的后续问题模板。
- `pattern_name` (`str`)：用于替换占位符的实际模式名称。

**返回值**：
- `str`：后续问题中占位符已被模式名称替换后的字符串。

**示例**：
```python
follow_up_question = "对于 PALCEHOLDER2，需要做哪些修改？"
pattern_name = "circle_pattern"
result = replace_pattern_name(follow_up_question, pattern_name)
# 结果: "对于 circle_pattern，需要做哪些修改？"
```

---

### 4. `extract_code_block`

**用途**：从给定文本中提取 Python 代码块，处理带有语言说明和不带语言说明的情况。

**签名**：
```python
def extract_code_block(text: str) -> str:
```

**参数**：
- `text` (`str`)：包含代码块的输入文本。

**返回值**：
- `str`：提取出的代码块内容，去除围栏符号。如果未找到代码块，则返回原始文本。

**示例**：
```python
text = """
这是您请求的代码：

```python
def hello_world():
    print("Hello, world!")
```
"""

code = extract_code_block(text)
# 结果:
# def hello_world():
#     print("Hello, world!")
```

---

### 5. `interpolate_and_resample`

**用途**：插值并重新采样一组原始点，以生成沿曲线均匀分布的 30 个关键帧。

**签名**：
```python
def interpolate_and_resample(original_points):
```

**参数**：
- `original_points` (`list`)：表示原始 2D 点的元组列表，例如 `[(x1, y1), (x2, y2), ...]`。

**返回值**：
- `list`：包含 30 个元组的列表，表示重新采样和缩放后的 2D 点。

**功能**：
1. 将原始点转换为 NumPy 数组。
2. 应用微小扰动以避免重复点。
3. 拟合带有最小平滑的 B 样条。
4. 在样条上生成细分的点。
5. 计算曲线的总长度并将其分为 29 个段。
6. 确定对应于段长度的均匀参数 `u` 值。
7. 在这些均匀 `u` 值处评估样条以获取重新采样的点。
8. 将点归一化并缩放到 `[-3.6, 3.6]` 范围内。

**示例**：
```python
original_points = [(0, 0), (1, 2), (2, 3), (4, 5)]
resampled_points = interpolate_and_resample(original_points)
# resampled_points 将包含沿曲线均匀分布的 30 个点。
```

---

### 6. `execute_code_and_compute_keyframes`

**用途**：执行提取的 Python 代码以生成曲线，采样点，归一化并缩放它们。

**签名**：
```python
def execute_code_and_compute_keyframes(extracted_code: str, log_file: str = "./log/model_interaction_log.txt") -> List[Tuple[int, Tuple[float, float]]]:
```

**参数**：
- `extracted_code` (`str`)：从 OpenAI 模型的响应中提取的 Python 代码，预期定义了 `shape_curve` 函数和 `t_range`。
- `log_file` (`str`, 可选)：用于记录交互日志的文件路径。默认为 `"./log/model_interaction_log.txt"`。

**返回值**：
- `List[Tuple[int, Tuple[float, float]]]`：一个元组列表，每个元组包含一个帧编号和一对缩放后的 `(x, y)` 坐标。

**功能**：
1. 记录提取的代码。
2. 在受控环境中执行代码，使 `numpy` 和 `math` 可用。
3. 从执行的代码中提取 `shape_curve` 函数和 `t_range`。
4. 从 `shape_curve` 根据 `t_range` 采样 30 个点。
5. 将采样的点归一化并缩放到 `[-3.6, 3.6]` 范围内。
6. 返回缩放后的点列表。

**示例**：
```python
extracted_code = """
def shape_curve(t):
    return (math.sin(t), math.cos(t))

t_range = (0, 2 * math.pi)
"""

keyframes = execute_code_and_compute_keyframes(extracted_code)
# keyframes 将包含表示圆形的 30 个缩放后的 (x, y) 点。
```

---

### 7. `interpolate_and_resample_velocity`

**用途**：插值并重新采样一组原始点，以在曲线上均匀分布的点处计算速度向量。

**签名**：
```python
def interpolate_and_resample_velocity(original_points):
```

**参数**：
- `original_points` (`list`)：表示原始 2D 点的元组列表。

**返回值**：
- `np.ndarray`：形状为 `(len(uniform_u), 2)` 的数组，表示每个重新采样点处的速度向量。

**功能**：
1. 类似于 `interpolate_and_resample`，但不是返回点，而是计算每个重新采样参数 `u` 处的第一导数（速度）。
2. 使用 B 样条插值并计算导数以获取切线向量。
3. 返回速度向量数组。

**示例**：
```python
original_points = [(0, 0), (1, 2), (2, 3), (4, 5)]
velocities = interpolate_and_resample_velocity(original_points)
# velocities 将包含对应于每个重新采样点的速度向量。
```

---

### 8. `interpolate_and_resample_velocity2`

**用途**：类似于 `interpolate_and_resample_velocity`，但允许指定输出点的数量，并使用速度向量重构点。

**签名**：
```python
def interpolate_and_resample_velocity2(original_points, num_output_points=200):
```

**参数**：
- `original_points` (`list`)：表示原始 2D 点的元组列表。
- `num_output_points` (`int`, 可选)：要重新采样的点数。默认为 `200`。

**返回值**：
- `tuple`：包含：
    - `reconstructed_points` (`list`)：重新采样和重构的 2D 点列表。
    - `velocity_vectors` (`list`)：连续点之间的速度向量列表。

**功能**：
1. 对原始点进行 B 样条插值。
2. 根据曲线长度将曲线重新采样为 `num_output_points` 个均匀分布的点。
3. 计算速度向量，即连续重新采样点之间的差值。
4. 使用速度向量从第一个点开始累积重构点。
5. 返回重构的点和速度向量。

**示例**：
```python
original_points = [(0, 0), (1, 2), (2, 3), (4, 5)]
reconstructed, velocities = interpolate_and_resample_velocity2(original_points, num_output_points=100)
# reconstructed 将包含 100 个重新采样的点。
# velocities 将包含这些点之间的 99 个速度向量。
```

---

### 9. `inside`

**用途**：核心函数，负责协调处理轨迹描述，与 OpenAI API 交互，执行生成的代码，插值速度，并保存结果。

**签名**：
```python
def inside(
    instruct_content: str,
    descriptions: List[str],
    follow_up_question_file: str = "./prompt/follow_up_question.txt",
    log_file: str = "./log/model_interaction_log.txt"
) -> str:
```

**参数**：
- `instruct_content` (`str`)：包含占位符的指令内容。
- `descriptions` (`List[str]`)：用于填充占位符的描述列表。
- `follow_up_question_file` (`str`, 可选)：包含后续问题的文件路径。默认为 `"./prompt/follow_up_question.txt"`。
- `log_file` (`str`, 可选)：用于记录交互和错误的日志文件路径。默认为 `"./log/model_interaction_log.txt"`。

**返回值**：
- `str`：根线速度的字符串表示（如果成功），或错误消息。

**功能**：
1. **占位符替换**：用实际描述替换 `instruct_content` 中的占位符。
2. **初始 OpenAI 交互**：
    - 将修改后的指令发送给 OpenAI API。
    - 记录用户指令和模型的响应。
3. **代码提取与执行**：
    - 从模型的响应中提取 Python 代码。
    - 执行代码以获取 `shape_curve` 和 `t_range`。
    - 从曲线中采样 30 个点。
4. **后续交互**：
    - 从文件中读取后续问题。
    - 用描述中提取的模式名称替换后续问题中的占位符。
    - 将后续问题发送给 OpenAI API。
    - 从响应中获取新的模式名称。
5. **速度插值与保存**：
    - 使用 `interpolate_and_resample_velocity2` 插值并重新采样速度。
    - 重塑速度数组并将其作为 `.npy` 文件保存在 `./npysave/` 目录中。
6. **错误处理**：
    - 记录在代码执行或插值过程中遇到的任何错误。
    - 如果初始代码执行失败，尝试将数学函数替换为 NumPy 等效函数并重试。
    - 如果错误依然存在，返回适当的错误消息。

**示例**：
```python
instruct_content = "生成基于 placeholder1 的模式。"
descriptions = ["圆形轨迹"]
result = inside(instruct_content, descriptions)
# result 将是根线速度的字符串表示或错误消息。
```

---

### 10. `log_results`

**用途**：将处理轨迹描述的结果附加到结果文件中。

**签名**：
```python
def log_results(result_file_path: str, descriptions: List[str], result: str):
```

**参数**：
- `result_file_path` (`str`)：用于记录结果的文件路径。
- `descriptions` (`List[str]`)：与结果对应的描述。
- `result` (`str`)：要记录的结果。

**返回值**：
- `None`

**功能**：
将结果附加到指定的 `result_file_path`，并用换行符分隔条目。

**示例**：
```python
log_results('./result/newResult.txt', ["圆形轨迹"], "处理成功。")
```

---

### 11. `main`

**用途**：脚本的入口点。通过读取指令模板和轨迹描述，处理每个轨迹并记录结果，管理整体工作流程。

**签名**：
```python
def main():
```

**参数**：
- 无

**返回值**：
- `None`

**功能**：
1. **文件路径设置**：
    - `instruct_file_path`：指令模板路径 (`./prompt/ChainPrompt.txt`)。
    - `trajectories_file_path`：轨迹描述路径 (`./prompt/trajectories.txt`)。
    - `result_file_path`：结果日志文件路径 (`./result/newResult.txt`)。
2. **读取文件**：
    - 从 `ChainPrompt.txt` 读取指令内容。
    - 从 `trajectories.txt` 读取轨迹描述。
3. **处理轨迹**：
    - 遍历每个轨迹描述。
    - 替换指令模板中的占位符为当前描述。
    - 调用 `inside` 函数处理指令和描述。
    - 将结果记录到 `newResult.txt`。
4. **环境设置**：
    - 在执行前将 `WORLD_SIZE` 环境变量设置为 `'1'`。

**示例**：
只需运行脚本将执行 `main` 函数。

```bash
python script.py
```

---

## 执行流程

1. **初始化**：
    - 从 `config.yaml` 加载 OpenAI API 密钥。
    - 设置 OpenAI 模型（默认为 `gpt-4o-mini`）。

2. **主工作流程（`main` 函数）**：
    - 从 `ChainPrompt.txt` 读取指令模板。
    - 从 `trajectories.txt` 读取轨迹描述。
    - 对于每个轨迹描述：
        - 替换指令模板中的占位符。
        - 调用 `inside` 函数处理指令和描述。
        - 将结果记录到 `newResult.txt`。
    - `inside` 函数处理与 OpenAI API 的交互、代码执行、插值、速度计算和结果保存。

3. **数据保存**：
    - 处理后的速度向量作为 `.npy` 文件保存在 `./npysave/` 目录中，文件名基于从 OpenAI 模型获得的模式名称。

4. **日志记录**：
    - 所有交互，包括用户指令、模型响应、提取的代码和错误，都记录在 `model_interaction_log.txt` 中。

---

## 错误处理

脚本包含健壮的错误处理机制：

1. **代码执行错误**：
    - 如果执行提取的代码失败，脚本会尝试将标准数学函数替换为其 NumPy 等效函数，并重试执行。
    - 如果错误依然存在，记录错误消息并停止处理该轨迹。

2. **插值错误**：
    - 在速度插值过程中发生的错误将被捕获、记录，并返回错误消息。

3. **日志记录**：
    - 所有错误都记录在 `model_interaction_log.txt` 中，附有描述性消息以便调试。

4. **优雅终止**：
    - 如果发生关键错误，脚本会返回信息性消息而不会崩溃，允许手动干预或重试。

---

## 使用示例

### 前提条件：

1. **配置文件**：确保存在 `config.yaml`，并包含正确的 OpenAI API 密钥。
2. **指令模板**：准备包含占位符（如 `placeholder1`、`placeholder2` 等）的 `ChainPrompt.txt`。
3. **轨迹描述**：在 `trajectories.txt` 中填写描述，每行一个。
4. **后续问题**：创建包含占位符 `PALCEHOLDER2` 的 `follow_up_question.txt`，用于后续提示。

### 目录结构：

```
project/
├── config.yaml
├── prompt/
│   ├── ChainPrompt.txt
│   ├── trajectories.txt
│   └── follow_up_question.txt
├── log/
│   └── model_interaction_log.txt
├── npysave/
│   └── # .npy 文件将保存在此目录
├── result/
│   └── newResult.txt
└── script.py
```

### 运行脚本：

导航到项目目录并执行：

```bash
python script.py
```

### 预期结果：

- 对于 `trajectories.txt` 中的每个描述，脚本将：
    - 替换指令模板中的占位符。
    - 与 OpenAI API 交互以生成并执行代码。
    - 进行插值并计算速度向量。
    - 将结果速度数据作为 `.npy` 文件保存在 `./npysave/` 目录中。
    - 在 `newResult.txt` 和 `model_interaction_log.txt` 中记录所有交互和结果。

---

## 注意事项

- **模型选择**：脚本默认使用 `gpt-4o-mini` 模型。要切换到其他模型（例如 `gpt-4o`），请取消注释脚本中的相应行。
  
- **日志记录**：确保 `log/` 目录存在并且可写，以便正确记录交互和错误。

- **数据保存**：确保 `npysave/` 目录存在并具有适当的写权限，以保存 `.npy` 文件。

- **错误消息**：脚本提供信息性错误消息以帮助排除故障。如果出现意外行为，请查看日志文件。

- **依赖项**：虽然导入了 `matplotlib.pyplot`，但在当前脚本中未使用。如果未来需要绘图，请确保正确使用或移除未使用的导入以清理代码。

---

# 结论

该 Python 脚本提供了一个全面的工作流程，用于处理轨迹描述，利用 OpenAI 的 GPT-4 能力生成并执行代码，并执行高级的插值和速度计算。通过详细的日志记录和错误处理，确保了脚本的健壮性和易于调试。正确的配置和遵循预期的目录结构将促进顺利的执行和数据管理。