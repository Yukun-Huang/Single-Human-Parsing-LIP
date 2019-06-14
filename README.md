# Single-Human-Parsing-LIP
A baseline model ( PSPNet ) for single-person human parsing task, training and evaluating on [Look into Person (LIP) dataset](http://sysu-hcp.net/lip/index.php).

## Model
The implementation of PSPNet is based on [HERE](https://github.com/Lextal/pspnet-pytorch).

## Dataset
To use our code, you should download [LIP dataset](http://sysu-hcp.net/lip/index.php) first.

Then, reorganize the dataset folder as below:

```
LIP
│ 
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
```

## Usage
python3  train.py  --data-path ~/myLIP
python3  eval.py  --data-path ~/myLIP [--visualize]

## Result

| model | overall acc. | mean acc. | mean IoU |
| :------: | :------: | :------: | :------: |
| resnet50 | 0.792 | 0.552 | 0.463 |
| densenet121 | 0.826 | 0.606 | 0.519 |
| squeezenet | 0.786 | 0.543 | 0.450 |

