## Workflow of the Scripts
1. `extract-connectivity.py`, extracts the given connectivity measure and creates a `.pkl` file
2. `inference-connectivity.py`, performs head selections and testing in unsupervised fashion and creates `.json` log file.
3. `create-dataset.py`, creates the dataset for logistic regression task.

## Result from Experiments
| algebraic connectivity | dataset | split | auroc | remarks | 
| :--- | :---: | :---: | :---: | :--- | 
| text - visual feidler's | radvqa | train | 0 | |
| text - viusal feidler's | radvqa | test | 0 | | 
| text fieder's | radvqa | train | 0 | |
| text fieder's | radvqa | test | 0 | |
| visual fieder's | radvqa | train | 0 | |
| visual fieder's | radvqa | test | 0 | |

## Effects of Number of Samples
- n = [10, 20, 50, 100, 200, 800, 1791]
- auroc = [0.5549, 0.5830, 0.5618, 0.5665, 0.6070, 0.5683, 0.5810, 0.5897]
[!Effects of number of training samples](./docs/train_samples-radvqa.png)

## Effects of Number of Max Heads
- n = [1, 5, 10, 20, 50, 100]
- auroc = [0.5201, 0.5458, 0.5864, 0.6037, 0.6070, 0.6070]
- selection = [1, 5, 10, 19, 25, 25]
