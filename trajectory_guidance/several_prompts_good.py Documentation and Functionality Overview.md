# several_prompts_good.py Documentation and Functionality Overview

## Overview

This Python script is designed to process trajectory descriptions, interact with the OpenAI GPT-4 model to generate and execute Python code for curve generation, interpolate and resample the resulting points, compute velocity vectors, and save the processed data for further use. The script is modular, with functions handling specific tasks such as placeholder replacement, code extraction, curve interpolation, and interaction logging.

### Key Functionalities:

1. **Placeholder Replacement**: Dynamically replaces placeholders in instruction templates with actual trajectory descriptions.
2. **Code Extraction and Execution**: Extracts Python code blocks from OpenAI's responses and executes them to obtain curve definitions.
3. **Curve Interpolation and Resampling**: Uses B-spline interpolation to smooth and resample curves into uniform points.
4. **Velocity Computation**: Calculates velocity vectors based on the resampled points.
5. **Data Logging and Saving**: Logs interactions and saves the processed velocity data in `.npy` format for later use.

---

## Table of Contents

1. [Dependencies](#dependencies)
2. [Configuration](#configuration)
3. [Functions](#functions)
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
4. [Execution Flow](#execution-flow)
5. [Error Handling](#error-handling)
6. [Usage Example](#usage-example)

---

## Dependencies

The script relies on the following Python libraries:

- **Standard Libraries**:
    - `os`
    - `math`
    - `re`
    - `typing`
- **Third-Party Libraries**:
    - `yaml`: For parsing YAML configuration files.
    - `numpy`: Numerical computations.
    - `matplotlib.pyplot`: Plotting (imported but not used in the script).
    - `scipy`: For interpolation, integration, and optimization.
    - `openai`: Interacting with OpenAI's GPT-4 model.

**Installation**:
Ensure all dependencies are installed. You can install them using `pip`:

```bash
pip install pyyaml numpy scipy openai matplotlib
```

---

## Configuration

The script expects a `config.yaml` file in the same directory, containing the OpenAI API key. The structure of `config.yaml` should be:

```yaml
token: YOUR_OPENAI_API_KEY
```

Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key.

---

## Functions

### 1. `replace_placeholders`

**Purpose**: Replaces placeholders in the instruction content with actual descriptions.

**Signature**:
```python
def replace_placeholders(instruct_content: str, descriptions: List[str]) -> str:
```

**Parameters**:
- `instruct_content` (`str`): The instruction template containing placeholders like `placeholder1`, `placeholder2`, etc.
- `descriptions` (`List[str]`): A list of descriptions to replace the placeholders.

**Returns**:
- `str`: The instruction content with placeholders replaced by corresponding descriptions.

**Example**:
```python
instruct_content = "Create a pattern based on placeholder1 and placeholder2."
descriptions = ["circle trajectory", "spiral motion"]
result = replace_placeholders(instruct_content, descriptions)
# Result: "Create a pattern based on circle trajectory and spiral motion."
```

---

### 2. `format_keyframes`

**Purpose**: Formats a list of keyframes into a structured string suitable for code insertion.

**Signature**:
```python
def format_keyframes(keyframes: list, pattern_name="origin_30points", keyframes_str="keyframes_str") -> str:
```

**Parameters**:
- `keyframes` (`list`): A list of keyframes, where each keyframe is a tuple in the form `(frame_number, (x_coordinate, y_coordinate))`.
- `pattern_name` (`str`, optional): The base name for the pattern. Defaults to `"origin_30points"`.
- `keyframes_str` (`str`, optional): An additional string to append to the pattern name. Defaults to `"keyframes_str"`.

**Returns**:
- `str`: A formatted string representing the keyframes in code.

**Example**:
```python
keyframes = [
    (0, (0.00, 0.00)),
    (1, (1.23, 4.56)),
    # ... more keyframes
]
formatted = format_keyframes(keyframes)
# Result:
# elif pattern == "origin_30points_keyframes_str":
#     kframes = [
#         (0, (0.00, 0.00)),
#         (1, (1.23, 4.56)),
#         ...
#     ]
```

---

### 3. `replace_pattern_name`

**Purpose**: Replaces a specific placeholder in the follow-up question with the actual pattern name.

**Signature**:
```python
def replace_pattern_name(follow_up_question: str, pattern_name: str) -> str:
```

**Parameters**:
- `follow_up_question` (`str`): The template follow-up question containing the placeholder `PALCEHOLDER2`.
- `pattern_name` (`str`): The actual pattern name to replace the placeholder.

**Returns**:
- `str`: The follow-up question with the placeholder replaced by the pattern name.

**Example**:
```python
follow_up_question = "What modifications are needed for PALCEHOLDER2?"
pattern_name = "circle_pattern"
result = replace_pattern_name(follow_up_question, pattern_name)
# Result: "What modifications are needed for circle_pattern?"
```

---

### 4. `extract_code_block`

**Purpose**: Extracts a Python code block from a given text, handling both fenced code blocks with language specification and without.

**Signature**:
```python
def extract_code_block(text: str) -> str:
```

**Parameters**:
- `text` (`str`): The input text containing the code block.

**Returns**:
- `str`: The extracted code block without the surrounding fences. If no code block is found, returns the original text.

**Example**:
```python
text = """
Here is the code you requested:

```python
def hello_world():
    print("Hello, world!")
```
"""

code = extract_code_block(text)
# Result:
# def hello_world():
#     print("Hello, world!")
```

---

### 5. `interpolate_and_resample`

**Purpose**: Interpolates and resamples a set of original points to generate 30 uniformly spaced keyframes along the curve.

**Signature**:
```python
def interpolate_and_resample(original_points):
```

**Parameters**:
- `original_points` (`list`): A list of tuples representing the original 2D points, e.g., `[(x1, y1), (x2, y2), ...]`.

**Returns**:
- `list`: A list of 30 tuples representing the resampled and scaled 2D points.

**Functionality**:
1. Converts the original points to NumPy arrays.
2. Applies a small perturbation to avoid duplicate points.
3. Fits a B-spline with minimal smoothing.
4. Generates a fine sampling of points along the spline.
5. Calculates the total length of the curve and divides it into 29 segments.
6. Determines uniform parameter `u` values corresponding to the segment lengths.
7. Evaluates the spline at these uniform `u` values to obtain resampled points.
8. Normalizes and scales the points to fit within the range `[-3.6, 3.6]`.

**Example**:
```python
original_points = [(0, 0), (1, 2), (2, 3), (4, 5)]
resampled_points = interpolate_and_resample(original_points)
# resampled_points will contain 30 uniformly spaced points along the curve.
```

---

### 6. `execute_code_and_compute_keyframes`

**Purpose**: Executes extracted Python code to generate a curve, samples points from it, normalizes and scales them.

**Signature**:
```python
def execute_code_and_compute_keyframes(extracted_code: str, log_file: str = "./log/model_interaction_log.txt") -> List[Tuple[int, Tuple[float, float]]]:
```

**Parameters**:
- `extracted_code` (`str`): The Python code extracted from the OpenAI model's response, expected to define a `shape_curve` function and `t_range`.
- `log_file` (`str`, optional): Path to the log file for recording interactions. Defaults to `"./log/model_interaction_log.txt"`.

**Returns**:
- `List[Tuple[int, Tuple[float, float]]]`: A list of tuples where each tuple contains a frame number and a pair of scaled `(x, y)` coordinates.

**Functionality**:
1. Logs the extracted code.
2. Executes the code in a controlled environment, making `numpy` and `math` available.
3. Extracts the `shape_curve` function and `t_range` from the executed code.
4. Samples 30 points from the curve defined by `shape_curve` over `t_range`.
5. Normalizes and scales the sampled points to fit within `[-3.6, 3.6]`.
6. Returns the list of scaled points.

**Example**:
```python
extracted_code = """
def shape_curve(t):
    return (math.sin(t), math.cos(t))

t_range = (0, 2 * math.pi)
"""

keyframes = execute_code_and_compute_keyframes(extracted_code)
# keyframes will contain 30 scaled (x, y) points representing a circle.
```

---

### 7. `interpolate_and_resample_velocity`

**Purpose**: Interpolates and resamples a set of original points to compute velocity vectors at uniformly spaced points along the curve.

**Signature**:
```python
def interpolate_and_resample_velocity(original_points):
```

**Parameters**:
- `original_points` (`list`): A list of tuples representing the original 2D points.

**Returns**:
- `np.ndarray`: An array of shape `(len(uniform_u), 2)` representing the velocity vectors at each resampled point.

**Functionality**:
1. Similar to `interpolate_and_resample`, but instead of returning points, it computes the first derivative (velocity) at each resampled `u` parameter.
2. Uses B-spline interpolation and calculates derivatives to obtain tangent vectors.
3. Returns the array of velocity vectors.

**Example**:
```python
original_points = [(0, 0), (1, 2), (2, 3), (4, 5)]
velocities = interpolate_and_resample_velocity(original_points)
# velocities will contain velocity vectors corresponding to each resampled point.
```

---

### 8. `interpolate_and_resample_velocity2`

**Purpose**: Similar to `interpolate_and_resample_velocity` but allows specifying the number of output points and reconstructs points using velocity vectors.

**Signature**:
```python
def interpolate_and_resample_velocity2(original_points, num_output_points=200):
```

**Parameters**:
- `original_points` (`list`): A list of tuples representing the original 2D points.
- `num_output_points` (`int`, optional): The number of points to resample. Defaults to `200`.

**Returns**:
- `tuple`: A tuple containing:
    - `reconstructed_points` (`list`): The resampled and reconstructed 2D points.
    - `velocity_vectors` (`list`): The list of velocity vectors between consecutive points.

**Functionality**:
1. Performs B-spline interpolation on the original points.
2. Resamples the curve into `num_output_points` uniformly spaced points based on curve length.
3. Computes velocity vectors as the difference between consecutive resampled points.
4. Reconstructs the points by cumulatively adding velocity vectors starting from the first point.
5. Returns both the reconstructed points and the velocity vectors.

**Example**:
```python
original_points = [(0, 0), (1, 2), (2, 3), (4, 5)]
reconstructed, velocities = interpolate_and_resample_velocity2(original_points, num_output_points=100)
# reconstructed will contain 100 resampled points.
# velocities will contain 99 velocity vectors between these points.
```

---

### 9. `inside`

**Purpose**: Core function that orchestrates the processing of trajectory descriptions, interacts with the OpenAI API, executes generated code, interpolates velocities, and saves the results.

**Signature**:
```python
def inside(
    instruct_content: str,
    descriptions: List[str],
    follow_up_question_file: str = "./prompt/follow_up_question.txt",
    log_file: str = "./log/model_interaction_log.txt"
) -> str:
```

**Parameters**:
- `instruct_content` (`str`): The instruction content with placeholders to be replaced.
- `descriptions` (`List[str]`): A list of descriptions to fill into the placeholders.
- `follow_up_question_file` (`str`, optional): Path to the file containing the follow-up question. Defaults to `"./prompt/follow_up_question.txt"`.
- `log_file` (`str`, optional): Path to the log file for recording interactions and errors. Defaults to `"./log/model_interaction_log.txt"`.

**Returns**:
- `str`: The root linear velocity as a string if successful, or an error message.

**Functionality**:
1. **Placeholder Replacement**: Replaces placeholders in `instruct_content` with actual descriptions.
2. **Initial OpenAI Interaction**:
    - Sends the modified instruction to the OpenAI API.
    - Logs the user instruction and the model's response.
3. **Code Extraction and Execution**:
    - Extracts Python code from the model's response.
    - Executes the code to obtain `shape_curve` and `t_range`.
    - Samples 30 points from the curve.
4. **Follow-Up Interaction**:
    - Reads a follow-up question from a file.
    - Replaces a pattern name placeholder with the actual pattern name extracted from descriptions.
    - Sends the follow-up question to the OpenAI API.
    - Retrieves a new pattern name from the response.
5. **Velocity Interpolation and Saving**:
    - Interpolates and resamples velocities using `interpolate_and_resample_velocity2`.
    - Reshapes the velocity array and saves it as a `.npy` file in the `./npysave/` directory.
6. **Error Handling**:
    - Logs any errors encountered during code execution or interpolation.
    - Attempts to replace math functions with NumPy equivalents if initial code execution fails.
    - Returns appropriate error messages if failures persist.

**Example**:
```python
instruct_content = "Generate a pattern based on placeholder1."
descriptions = ["circle trajectory"]
result = inside(instruct_content, descriptions)
# result will be the root linear velocity as a string or an error message.
```

---

### 10. `log_results`

**Purpose**: Appends the result of processing a trajectory description to a result file.

**Signature**:
```python
def log_results(result_file_path: str, descriptions: List[str], result: str):
```

**Parameters**:
- `result_file_path` (`str`): Path to the file where results are logged.
- `descriptions` (`List[str]`): The descriptions corresponding to the result.
- `result` (`str`): The result to be logged.

**Returns**:
- `None`

**Functionality**:
Appends the result to the specified `result_file_path`, separating entries with newline characters.

**Example**:
```python
log_results('./result/newResult.txt', ["circle trajectory"], "Processed successfully.")
```

---

### 11. `main`

**Purpose**: Entry point of the script. Manages the overall workflow by reading instruction templates and trajectory descriptions, processing each trajectory, and logging results.

**Signature**:
```python
def main():
```

**Parameters**:
- None

**Returns**:
- `None`

**Functionality**:
1. **File Paths Setup**:
    - `instruct_file_path`: Path to the instruction template (`./prompt/ChainPrompt.txt`).
    - `trajectories_file_path`: Path to the trajectory descriptions (`./prompt/trajectories.txt`).
    - `result_file_path`: Path to the result log file (`./result/newResult.txt`).
2. **Reading Files**:
    - Reads the instruction content from `ChainPrompt.txt`.
    - Reads trajectory descriptions from `trajectories.txt`.
3. **Processing Trajectories**:
    - Iterates over each trajectory description.
    - Replaces placeholders in the instruction content with the current description.
    - Calls the `inside` function to process the instruction and description.
    - Logs the result to `newResult.txt`.
4. **Environment Setup**:
    - Sets the `WORLD_SIZE` environment variable to `'1'` before execution.

**Example**:
Simply running the script will execute the `main` function.

```bash
python script.py
```

---

## Execution Flow

1. **Initialization**:
    - Load the OpenAI API key from `config.yaml`.
    - Set the OpenAI model (`gpt-4o-mini` by default).

2. **Main Workflow (`main` Function)**:
    - Read the instruction template from `ChainPrompt.txt`.
    - Read trajectory descriptions from `trajectories.txt`.
    - For each trajectory description:
        - Replace placeholders in the instruction template.
        - Invoke the `inside` function to process the instruction and description.
        - Log the result to `newResult.txt`.
    - The `inside` function handles interactions with the OpenAI API, code execution, interpolation, velocity computation, and saving of results.

3. **Data Saving**:
    - Processed velocity vectors are saved as `.npy` files in the `./npysave/` directory, named based on the pattern name obtained from the OpenAI model.

4. **Logging**:
    - All interactions, including user instructions, model responses, extracted code, and errors, are logged to `model_interaction_log.txt`.

---

## Error Handling

The script includes robust error handling mechanisms:

1. **Code Execution Errors**:
    - If executing the extracted code fails, the script attempts to replace standard math functions with their NumPy equivalents and retries execution.
    - If errors persist, an error message is logged, and processing for that trajectory is halted.

2. **Interpolation Errors**:
    - Errors during velocity interpolation are caught, logged, and result in an error message being returned.

3. **Logging**:
    - All errors are recorded in `model_interaction_log.txt` with descriptive messages to facilitate debugging.

4. **Graceful Termination**:
    - If critical errors occur, the script returns informative messages without crashing, allowing for manual intervention or retries.

---

## Usage Example

### Prerequisites:

1. **Configuration File**: Ensure `config.yaml` is present with the correct OpenAI API key.
2. **Instruction Template**: Prepare `ChainPrompt.txt` with placeholders like `placeholder1`, `placeholder2`, etc.
3. **Trajectory Descriptions**: Populate `trajectories.txt` with descriptions, each on a new line.
4. **Follow-Up Question**: Create `follow_up_question.txt` with the appropriate follow-up prompt containing the placeholder `PALCEHOLDER2`.

### Directory Structure:

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
│   └── # .npy files will be saved here
├── result/
│   └── newResult.txt
└── script.py
```

### Running the Script:

Navigate to the project directory and execute:

```bash
python script.py
```

### Expected Outcome:

- For each description in `trajectories.txt`, the script will:
    - Replace placeholders in the instruction template.
    - Interact with the OpenAI API to generate and execute code.
    - Interpolate and compute velocity vectors.
    - Save the resulting velocity data in the `./npysave/` directory as `.npy` files.
    - Log all interactions and results in `newResult.txt` and `model_interaction_log.txt`.

---

## Notes

- **Model Selection**: The script uses the `gpt-4o-mini` model by default. To switch to a different model (e.g., `gpt-4o`), uncomment the corresponding line in the script.
  
- **Logging**: Ensure that the `log/` directory exists and is writable to allow proper logging of interactions and errors.

- **Data Saving**: Ensure that the `npysave/` directory exists and has appropriate write permissions to save the `.npy` files.

- **Error Messages**: The script provides informative error messages to aid in troubleshooting. Review the log files if unexpected behavior occurs.

- **Dependencies**: While `matplotlib.pyplot` is imported, it is not utilized in the current script. If plotting is required in the future, ensure proper usage or remove the unused import to clean up the code.

---

# Conclusion

This Python script provides a comprehensive workflow for processing trajectory descriptions, leveraging OpenAI's GPT-4 capabilities to generate and execute code, and performing advanced interpolation and velocity computations. With detailed logging and error handling, it ensures robustness and ease of debugging. Proper configuration and adherence to the expected directory structure will facilitate smooth execution and data management.