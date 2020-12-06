# Single-Human-Parsing-LIP
PSPNet implemented in PyTorch for **single-person human parsing** task, evaluating on Look Into Person (LIP) dataset.

## Model
The implementation of PSPNet is based on [HERE](https://github.com/Lextal/pspnet-pytorch).

Trained model weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/13DzOvUoIx0JR-BTEilhLqdAIp3h0H5Zj) or [Baidu Drive](https://pan.baidu.com/s/1SuGbwL1CF7pLxN1olBc49Q) (提取码：43cu).

## Environment
* Python 3.6
* PyTorch == 1.1.0
* torchvision == 0.3.0
* matplotlib

## Dataset
To use our code, firstly you should download LIP dataset from [HERE](http://sysu-hcp.net/lip/index.php).

Then, reorganize the dataset folder as below:

```
myLIP
│ 
└───train
│   │   id.txt
│   │
│   └───image
│   │   │   77_471474.jpg
│   │   │   113_1207747.jpg
│   │   │   ...
│   │
│   └───gt
│   │   │   77_471474.png
│   │   │   113_1207747.png
│   │   │   ...
│
└───val
│   │   id.txt
│   │
│   └───image
│   │   │   100034_483681.jpg
│   │   │   10005_205677.jpg
│   │   │   ...
│   │
│   └───gt
│   │   │   100034_483681.png
│   │   │   10005_205677.png
│   │   │   ...
│
└───test
│   │   id.txt
│   │
│   └───image
│   │   │   100012_501646.jpg
│   │   │   ...
```

## Usage
```
python3  train.py  --data-path PATH-TO-LIP  --backend [resnet50 | densenet | squeezenet]

python3  eval.py  --data-path PATH-TO-LIP  --backend [resnet50 | densenet | squeezenet]  [--visualize]

python3  inference.py  demo/test.jpg  --backend [resnet50 | densenet | squeezenet]
```

## Evaluation

| model | overall acc. | mean acc. | mean IoU |
| :------: | :------: | :------: | :------: |
| resnet50 | 0.792 | 0.552 | 0.463 |
| resnet101 | 0.805 | 0.579 | 0.489 |
| densenet121 | 0.826 | 0.606 | 0.519 |
| squeezenet | 0.786 | 0.543 | 0.450 |

## Visualization

```
>> python3  eval.py  --data-path PATH-TO-LIP  --visualize
```
<img src="https://github.com/hyk1996/Single-Human-Parsing-LIP/blob/master/demo/demo.jpg"  height="250" width="500">

```
>> python3  inference.py  demo/test.jpg
```
<img src="https://github.com/hyk1996/Single-Human-Parsing-LIP/blob/master/demo/result.jpg"  height="250" width="500">
