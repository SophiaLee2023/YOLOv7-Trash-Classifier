import os, torch, torchvision, torchaudio

def run_in_terminal(command_list: list) -> None:
    for command_str in command_list:
        os.system(command_str)

def install_requirements(GPU_integration: bool = False) -> None:
    run_in_terminal([
        "pip install -r requirements.txt", # downloads required packages
        "python TACO/download_script.py" # downloads TACO images (NOTE: FCPS wifi results in a connection error) and generates annotations
    ])

    if GPU_integration: # NOTE: for GPU-PyTorch integration (optional), install CUDA-11.7.0 and cuDNN-8.6.0
        run_in_terminal([ 
            "pip uninstall torch", # remove torch-1.13.0+cpu (default installation is CPU only)
            "pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio===0.13.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html",
        ])
# install_requirements(True)

print(f"PyTorch: {torch.__version__}\t TorchVision: {torchvision.__version__}\t TorchAudio: {torchaudio.__version__}\n" +\
      f"CUDA enabled: {torch.cuda.is_available()}\t CuDNN enabled: {torch.backends.cudnn.enabled}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

run_in_terminal([ # custom TACO-based model
    "python train.py --workers 8 --device 0 --epochs 100 --data data/TACO.yaml --cfg cfg/training/yolov7-tiny_nc60.yaml --weights " " --name trash_classifier --cache-images",
    "python detect.py --img 1280 --weights runs/train/trash_classifier/weights/best.pt --source TACO/images/test --conf 0.1 --name trash_classifier"
    # f"python detect.py --img 1280 --weights runs/train/trash_classifier/weights/best.pt --source 0 --conf 0.1 --name webcam_recording"
])