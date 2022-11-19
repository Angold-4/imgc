# lossyrnn

**Pytorch Implementation of the [paper](https://arxiv.org/abs/1511.06085) "Variable Rate Image Compression With Recurrent Neural Network".**

## Abstact

## Prerequisites for Code

### Windows

#### install python3.10 and create virtual environment
Please follow the [official guide](https://www.python.org/downloads/windows/) to install the python 3.10.

After you clone the repository, create virtual environment ([venv](https://docs.python.org/3/library/venv.html)).
 
```
cd imgc\lossyrnn
py -m venv .lossyrnn
```
Then, enter the virtual environment
```
.lossyrnn\Scripts\activate.bat
```


#### install pytorch with cuda
Please follow the [offiicial guide](https://pytorch.org/get-started/locally/) to install the cuda version of pytorch.
```
py -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
py -m pip install -r requirements.txt
```

#### without cuda support
```
py -m pip install -r requirements.txt
```

## Quick Start

### Training 

If you want to train this model on your own machine, I made a copy of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset [here](https://drive.google.com/file/d/1kWt_JKkkR1ORckDixkVmAkA4miY7R-GA/view?usp=share_link). (If you don't have a GPU, remove `--cuda`)

```
python3 train.py --batch-size 32 --cuda -f <dataset path>
```




### Evaluation
