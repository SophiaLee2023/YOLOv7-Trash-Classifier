# YOLO v7 TACO Trash Classifier

## Set-up Commands

``` shell
# downloads required packages
pip install -r requirements.txt

# downloads TACO images (NOTE: FCPS wifi results in a connection error) and generates annotations
python TACO/download_script.py

# optional: reinstall the GPU-compatible version of PyTorch to make training faster
pip uninstall torch
pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio===0.13.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
```

## Train.py and Detect.py

``` shell
# training command
python train.py --workers 8 --device 0 --epochs 100 --data data/TACO.yaml --cfg cfg/training/yolov7-tiny_nc60.yaml --weights " " --name trash_classifier --cache-images

# detection command from two different sources (check results in runs/detect/trash_classifier)
python detect.py --img 1280 --weights runs/train/trash_classifier/weights/best.pt --source TACO/images/test --conf 0.1 --name trash_classifier
python detect.py --img 1280 --weights runs/train/trash_classifier/weights/best.pt --source 0 --conf 0.1 --name webcam_recording
```

If using pre-trained weights, change the file path after "--weights to the location of the weights.pt file.