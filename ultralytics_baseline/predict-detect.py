from ultralytics import YOLO
# Load a model

# model = YOLO(r"/projects/SuperResolutionData/carolinali-shadowRemoval/baseline_updated_result/weights/best.pt",task='detect')  # load a pretrained model (recommended for training)
model = YOLO(r"/projects/SuperResolutionData/carolinali-shadowRemoval/log_chroma_weights/best.pt",task='detect')


# model.predict(source = r"/projects/SuperResolutionData/carolinali-shadowRemoval/baseline/jpgs", imgsz=1280, save=True)
model.predict(source = r"/projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_8_bit_png", imgsz=1280, save=True)
