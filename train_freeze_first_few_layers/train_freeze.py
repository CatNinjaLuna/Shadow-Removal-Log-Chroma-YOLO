import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
import argparse
import torch
import torch.nn as nn
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def set_trainable_first_n_layers(yolo_model, n):
    m = yolo_model.model
    layers = None
    if hasattr(m, 'model') and isinstance(getattr(m, 'model', None), nn.ModuleList):
        layers = list(m.model)
    else:
        layers = list(m.children())
    for i, layer in enumerate(layers):
        trainable = i < max(0, int(n))
        for p in layer.parameters():
            p.requires_grad = trainable


def set_eval_for_frozen_layers(yolo_model, freeze_indices):
    m = yolo_model.model
    layers = None
    if hasattr(m, 'model') and isinstance(getattr(m, 'model', None), nn.ModuleList):
        layers = list(m.model)
    else:
        layers = list(m.children())
    for i in freeze_indices:
        if 0 <= i < len(layers):
            layers[i].eval()


def summarize_trainable_and_bn(model, tag):
    tp = 0
    fp = 0
    for p in model.model.parameters():
        if getattr(p, 'requires_grad', False):
            tp += p.numel()
        else:
            fp += p.numel()
    bn_total = 0
    bn_eval = 0
    bn_train = 0
    for mod in model.model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            bn_total += 1
            if getattr(mod, 'training', True):
                bn_train += 1
            else:
                bn_eval += 1
    print(f"[{tag}] trainable_params={tp} frozen_params={fp} bn_total={bn_total} bn_eval={bn_eval} bn_train={bn_train}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=r'./yolo11s.pt')
    parser.add_argument('--data', type=str, default=r'/hy-tmp/data/data.yaml')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--close_mosaic', type=int, default=10)
    parser.add_argument('--workers', type=int, default=40)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--epochs1', type=int, default=50)
    parser.add_argument('--epochs2', type=int, default=250)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name1', type=str, default='exp_stage1')
    parser.add_argument('--name2', type=str, default='exp_stage2')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = YOLO(args.model)
    set_trainable_first_n_layers(model, args.layers)
    total_layers = 0
    try:
        total_layers = len(list(model.model.model))
    except Exception:
        total_layers = len(list(model.model.children()))
    n = max(0, int(args.layers))
    n = min(n, total_layers)
    freeze_indices = [i for i in range(n, total_layers)]
    set_eval_for_frozen_layers(model, freeze_indices)
    def _make_eval_cb(indices):
        def _cb(trainer):
            try:
                set_eval_for_frozen_layers(trainer.model, indices)
            except Exception:
                pass
        return _cb
    model.add_callback('on_train_start', _make_eval_cb(freeze_indices))
    model.add_callback('on_train_epoch_start', _make_eval_cb(freeze_indices))
    print(f"[stage1] freeze_indices={freeze_indices}")
    summarize_trainable_and_bn(model, 'stage1_before_train')
    model.train(
                data=args.data,
                imgsz=args.imgsz,
                epochs=args.epochs1,
                single_cls=False,
                batch=args.batch,
                close_mosaic=args.close_mosaic,
                workers=args.workers,
                device=args.device,
                optimizer=args.optimizer,
                amp=args.amp,
                project=args.project,
                name=args.name1,
                freeze=freeze_indices,
                )
    best = None
    try:
        best = getattr(getattr(model, 'trainer', None), 'best', None)
    except Exception:
        best = None
    if not best:
        sd = getattr(getattr(model, 'trainer', None), 'save_dir', None)
        if sd:
            p = os.path.join(sd, 'weights', 'best.pt')
            if os.path.exists(p):
                best = p
    if not best:
        best = args.model
    model2 = YOLO(best)
    for p in model2.model.parameters():
        p.requires_grad = True
    summarize_trainable_and_bn(model2, 'stage2_before_train')
    model2.train(
                data=args.data,
                imgsz=args.imgsz,
                epochs=args.epochs2,
                single_cls=False,
                batch=args.batch,
                close_mosaic=args.close_mosaic,
                workers=args.workers,
                device=args.device,
                optimizer=args.optimizer,
                amp=args.amp,
                project=args.project,
                name=args.name2,
                )


os.system('shutdown')
