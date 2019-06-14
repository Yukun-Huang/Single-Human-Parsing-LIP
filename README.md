# Single-Human-Parsing-LIP
A baseline model ( PSPNet ) for single-person human parsing task, training and evaluating on Look into Person (LIP) dataset.

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
│   │   │   file111.txt
│   │   │   file112.txt
│   │   │   ...
│
└───folder2
    │   file021.txt
    │   file022.txt
```

&ensp;&ensp;  ~/dataset/DukeMTMC-reID/  
&ensp;&ensp;  ~/dataset/DukeMTMC-reID/bounding_box_test/  
&ensp;&ensp;  ~/dataset/DukeMTMC-reID/bounding_box_train/  
&ensp;&ensp;  ~/dataset/DukeMTMC-reID/query/  
&ensp;&ensp;  ~/dataset/DukeMTMC-reID/attribute/  
&ensp;&ensp;  ~/dataset/DukeMTMC-reID/attribute/duke_attribute.mat  


## Usage
python3  train.py  --data-path  ~/dataset  --dataset  [market | duke]  --model  resnet50  
python3  test.py  --data-path  ~/dataset  --dataset  [market | duke]  --model  resnet50  


## Result

| model | overall acc. | mean acc. | mean IoU |
| :------: | :------: | :------: | :------: |
| resnet50 | 0.792 | 0.552 | 0.463 |
| densenet121 | 0.826 | 0.606 | 0.519 |
| squeezenet | 0.786 | 0.543 | 0.450 |

