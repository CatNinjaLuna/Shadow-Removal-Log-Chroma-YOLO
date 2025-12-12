'''
# (optional) activate virtual env
conda init bash
source ~/.bashrc
conda activate SR-shadow-removal

'''

# train.py

from ultralytics import YOLO

#model = YOLO(r"/projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange/runs/train/fuse_pt/weights/best.pt")  
#model = YOLO(r"/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics-src-new-log-chroma/runs/train/exp2/weights/best.pt") 
model = YOLO(r"/projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange/runs/train/fuse_pt/weights/best.pt")

model.val(data=r"/projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange/fuse_log_chroma_data/fuse_log_chroma_data.yaml",
            imgsz=1280, 
            epochs=300,  
            batch=64,  
            device=0,  
            workers=0,  
            pretrained=True,
            save=True,
            plots=True,
            plot_title_prefix="Y_Channel_Exchanged_")  

# This file must reside under a file path that is similar to: /projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics_baseline/val_rename.py
# where the project root has a subfolder "ultralytics", that contains all the pretrained yolo definitions
# /projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics_baseline/ultralytics

'''
Purpose of this file:
1. Rename validation dataset filenames
2. Align them to the new fused / Y-channel-exchanged images
3. Ensure YOLO sees the correct val split during model.val(...)
'''
