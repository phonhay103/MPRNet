
<!-- ## Training
- Download the [Datasets](Datasets/README.md)

- Train the model with default arguments by running

```
python train.py
```
-->

## Pre-requisites
The project was developed using python 3 with the following packages.
- Pytorch
- Opencv
- Numpy
- Scikit-image
- Pillow

1. Install [Pytorch](https://pytorch.org/get-started/locally/)
2. Install with pip:
```bash
pip install -r requirements.txt
```

## Datasets
- Rain 13k - Test: [Here](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs)
- Place it in `datasets`

## Evaluation
```
python test.py
```
or
```
python test.py --weights <model_weights> --input_dir <input_path> --result_dir <result_path>
```