# Single-Human-Parsing-LIP
A baseline model ( PSPNet ) for single-person human parsing task, training and testing on Look into Person  dataset.

## Model
We built model with PyTorch 0.4.1 and the implementation of PSPNet was based on [Here](https://github.com/Lextal/pspnet-pytorch).

Trained model weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/13DzOvUoIx0JR-BTEilhLqdAIp3h0H5Zj) or [Baidu Drive](https://pan.baidu.com/s/1SuGbwL1CF7pLxN1olBc49Q) (提取码：43cu).

## Dataset
To use our code, firstly you should download LIP dataset from [Here](http://sysu-hcp.net/lip/index.php).

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
python3  train.py  --data-path ~/myLIP

python3  eval.py  --data-path ~/myLIP [--visualize]

python3  inference.py  demo/test_a.jpg
```

## Result

### Evaluation

| model | overall acc. | mean acc. | mean IoU |
| :------: | :------: | :------: | :------: |
| resnet50 | 0.792 | 0.552 | 0.463 |
| densenet121 | 0.826 | 0.606 | 0.519 |
| squeezenet | 0.786 | 0.543 | 0.450 |

### Visualization

![demo](https://github.com/hyk1996/Single-Human-Parsing-LIP/raw/master/demo/demo.jpg)

```
>> python3  inference.py  demo/test_a.jpg
```
![demo](https://github.com/hyk1996/Single-Human-Parsing-LIP/raw/master/demo/result.jpg)
