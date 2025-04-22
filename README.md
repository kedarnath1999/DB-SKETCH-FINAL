# WM-Sketch Classification

This project implements several variants of logistic regression models optimized for data streams and high-dimensional sparse datasets.

## Methods Implemented

1. **Uncompressed Logistic Regression**
   - Traditional logistic regression model that maintains full weights for each feature.

2. **WM-Sketch**
   - Uses a sketching technique to compress the weight vector and update it in a space-efficient way.
   - Tracks top‑K important features using a heap.

3. **AWM-Sketch (Active-Set Weight Median Sketch)**
   - Hybrid model that keeps exact weights for top‑K features and sketches the rest.
   - Offers a balance between compression and accuracy.

4. **Truncated Model**
   - Stores only top‑K feature weights explicitly.
   - All other weights are treated as zero.

## Requirements

- Java 8 or higher
- `json.jar` (for JSON output)
- `jfreechart-1.5.0.jar` and `jcommon-1.0.23.jar` (for graph visualization)

## Compiling

In terminal:

```bash
javac -cp json.jar:. WMSketchClassification.java
```

## Running the Classifier

To run with a training file:

```bash
java -cp json.jar:. WMSketchClassification --train=./data/rcv1_test.binary --method=AWMsketch
```


## Data Format

Your training and test datasets must follow LIBSVM format:

```
<label> <feature1_index>:<feature1_value> <feature2_index>:<feature2_value> ...
```

## Execution Examples

- **Uncompressed Logistic Regression**:
  ```bash
  java -cp json.jar:. WMSketchClassification --train=./data/rcv1_test.binary --method=UncompressedLogisticRegression
  ```

- **WM-Sketch**:
  ```bash
  java -cp json.jar:. WMSketchClassification --train=./data/rcv1_test.binary --method=WMSketch
  ```

- **Active-Set WM-Sketch**:
  ```bash
  java -cp json.jar:. WMSketchClassification --train=./data/rcv1_test.binary --method=AWMsketch
  ```

- **Truncated Model**:
  ```bash
  java -cp json.jar:. WMSketchClassification --train=./data/rcv1_test.binary --method=TruncatedModel
  ```




## Graph Comparison
### Compile

```bash
javac -cp .:json.jar:jfreechart-1.5.0.jar:jcommon-1.0.23.jar CompareByMemoryBudget.java
javac -cp .:json.jar:jfreechart-1.5.0.jar:jcommon-1.0.23.jar CompareReconstructionError.java
```

### Run

```bash
java -cp .:json.jar:jfreechart-1.5.0.jar:jcommon-1.0.23.jar CompareByMemoryBudget
java -cp .:json.jar:jfreechart-1.5.0.jar:jcommon-1.0.23.jar CompareReconstructionError
```

## Additional Notes

- Use `--topk=<int>` to select how many top features are tracked.
- `--lr_init` and `--l2_reg` can be used to tune learning parameters.

