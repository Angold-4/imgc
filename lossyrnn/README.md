# lossyrnn

**Pytorch Implementation of the [paper](https://arxiv.org/abs/1511.06085) "Variable Rate Image Compression With Recurrent Neural Network".**

## 1. Abstact

## 2. Prerequisites for Code

### i. install python3.10

Please follow the [official guide](https://www.python.org/downloads/windows/) to install the python 3.10. first.
After you clone the repository, create virtual environment ([venv](https://docs.python.org/3/library/venv.html)).

### ii. Create virtual environment

#### Windows

```
cd imgc\lossyrnn
py -m venv .lossyrnn
```
Then, enter the virtual environment
```
.lossyrnn\Scripts\activate.bat
```

#### Unix-like Systems

```
cd imgc/lossyrnn
py -m venv .lossyrnn
```
Then, enter the virtual environment
```
.lossyrnn/bin/activate
```


### iii. install pytorch with cuda
Please follow the [offiicial guide](https://pytorch.org/get-started/locally/) to install the cuda version of pytorch.
```
py -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
py -m pip install -r requirements.txt
```

## 3. Quick Start

### i. Training 

If you want to train this model on your own machine, I made a copy of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset [here](https://drive.google.com/file/d/1kWt_JKkkR1ORckDixkVmAkA4miY7R-GA/view?usp=share_link). (If you don't have a GPU, remove `--cuda`)

#### With cuda support

```
python3 train.py --batch-size 32 --cuda -f <dataset path>
```
(remove the `--cuda` label if you want to disable the gpu)

### ii. Evaluation

#### Encode & Decode
**Using your trained model to compress any image:**

* Encode
```
python3 encoder.py --model <encoder.pth path> --input <input path> --output <output path>
```
* Decode
```
python3 decoder.py --model <decoder.pth path> --input <input.npz path> --out <output path>
```

#### Experiment (OSX Only)
**Evaluate the performance of your trained model by comparing the [SSIM](https://en.wikipedia.org/wiki/Structural_similarity) with jpeg2000.**

1. Get testing kodak images:
```sh
./experiment/get_kodak.sh
```

2. Compress them using your model
```sh
./experiment/enc_dec.sh
```

3. Compress them using jpeg2000 (make sure you've installed imagemagick)
```sh
./experiment/jpeg.sh
```

4. Comparing them by running SSIM scripts.
```sh
./experiment/ssim.sh
```

