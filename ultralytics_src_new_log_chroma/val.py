'''
# (optional) activate virtual env
conda init bash
source ~/.bashrc
conda activate SR-shadow-removal

'''

# train.py

from ultralytics import YOLO

model = YOLO(r"/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics-src-new-log-chroma/runs/train/exp2/weights/best.pt") 


model.val(data="D:\D-V8-G\\ultralytics-mainG\\ultralytics-main\D-data.yaml", #??????????
            imgsz=640,  # 输入图像的大小为整数或 w,h
            epochs=300,  # 要训练的次数
            batch=64,  # 每批次的图像数量（AutoBatch 为 -1）
            device=0,  # 要运行的设备，即 cuda device=0 或 device=0,1,2,3 或 device=cpu
            workers=0,  # 用于数据加载的工作线程数（如果是 DDP，则为每个 RANK）
            pretrained=True,
            save=True)  # True的时候则从上一个检查点恢复训练