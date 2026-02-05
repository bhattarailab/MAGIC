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

## Results of Logistic Regression
```bash
python inference-connectivity.py --train "./outputs/feidlermst_all_train_nonormalized.pkl" --test "./outputs/feidlermst_all_test_nonormalized.pkl" --name feidlermst_nonormalized --type fiedler_value --n-train 200 -l

python logistic-regression.py --csv "./outputs/datasets/feidlermst_nonormalized.csv"  --epoch 500 

selected heads only
train auroc = 0.6240951206611991, train_loss = 0.6865625381469727
test auroc = 0.5910433652369136, test_loss = 0.686627209186554
```

```bash
Not fused all heads
train auroc = 0.7077096356904802, train_loss = 0.6645540595054626
test auroc = 0.6777738390641616, test_loss = 0.6676965951919556

Fused all heads
train auroc = 0.7063937372153932, train_loss = 0.6649812459945679
test auroc = 0.6780298554492102, test_loss = 0.6677396297454834

```

```bash
Unsupervised = AUC score 0.5864744574421995

Supervised consdiering selected heads only
	train auroc = 0.5652564568238735, train_loss = 0.6888082027435303
	test auroc = 0.5351037851037851, test_loss = 0.6886552572250366


Supervised considering selected heads only with fused measure
	train auroc = 0.544981902413224, train_loss = 0.68860924243927
	test auroc = 0.48739611642837444, test_loss = 0.6888160705566406

Supervised considering all heads
	train auroc = 0.7006601795739421, train_loss = 0.67869633436203
	test auroc = 0.6606404348339832, test_loss = 0.6799866557121277

Supervised consdierign all heads with fused measure
	train auroc = 0.700791610111708, train_loss = 0.6796728372573853
	test auroc = 0.6664697309858598, test_loss = 0.6804147958755493
```

```bash
max_attentions 

Supervised selected heads not fuse
	train auroc = 0.6439952971763943, train_loss = 0.6640625596046448
	test auroc = 0.6241088660443499, test_loss = 0.6772207617759705

Supervised all heads fused
	train auroc = 0.7899580696757091, train_loss = 0.5883111953735352
	test auroc = 0.7203316396864784, test_loss = 0.648457407951355
```

## Results from All Basics
| metric | unsupervised | no fusion | arithmetic fusion | divide fusion |
|--|--|--|--|--|
| max attention | - | 0.72169 | 0.72809 | 0.6811 | 
| mean attention | 0.575 | 0.66402 | 0.66359 | 0.68181 | 
| entropy | 0.5044 | 0.70528 | 0.68202 | 0.68297 | 
| gini | 0.5806 | 0.73356 | 0.72612 | 0.6878 | 
| effective rank | 0.5149 | 0.61508 | 0.61115 | 0.68088 | 

```bash
# with fusion as text - vision
mean attention = unsupervised 0.575 to 0.66359
entropy = unsupervised 0.5044 to 0.68202
gini = unsupervised 0.5806 to 0.72612
effective_rank = unsupervised 0.5149 to 0.61115
```