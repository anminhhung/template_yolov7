## Requirment
``` shell
    pip install -r requirements.txt
```

## Training

Run training

``` shell
# train p5 models
python train.py --epochs 1 --workers 4 --device 0 --batch-size 15 --data human_detection_dataset/data.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights "" --name yolov7 --hyp cfg/data/hyp.scratch.p5.yaml

# train p6 models
python train_aux.py  --epochs 1 --workers 8 --device 0 --batch-size 16 --data human_detection_dataset/data.yaml --img 640 640 --cfg cfg/training/yolov7-w6.yaml --weights "" --name yolov7 --hyp cfg/data/hyp.scratch.p6.yaml


```

## Reference
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)