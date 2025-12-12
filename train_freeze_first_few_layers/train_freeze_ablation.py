import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
import argparse
import torch
import torch.nn as nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_frozen_first_n_layers(yolo_model, n):
    """
    Freeze the first n layers (0..n-1) and make the rest trainable.
    This is the opposite of set_trainable_first_n_layers.
    """
    m = yolo_model.model
    if hasattr(m, 'model') and isinstance(getattr(m, 'model', None), nn.ModuleList):
        layers = list(m.model)
    else:
        layers = list(m.children())

    for i, layer in enumerate(layers):
        trainable = i >= max(0, int(n))  # i < n -> frozen, i >= n -> trainable
        for p in layer.parameters():
            p.requires_grad = trainable


def set_eval_for_frozen_layers(yolo_model, freeze_indices):
    """
    Put frozen layers into eval() mode so their BatchNorm stats don't update.
    """
    m = yolo_model.model
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
    print(f"[{tag}] trainable_params={tp} frozen_params={fp} "
          f"bn_total={bn_total} bn_eval={bn_eval} bn_train={bn_train}")


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
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--freeze_list', type=str, default='3,4,5,6',
                        help='comma-separated list of number of frozen layers, '
                             'e.g., "3,4,5,6" = freeze 1-3, 1-4, 1-5, 1-6')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # parse freeze configuration list
    freeze_list = [int(x) for x in args.freeze_list.split(',') if x.strip()]

    for n in freeze_list:
        print(f"\n==============================")
        print(f"Training with layers 1–{n} frozen (indices 0..{n-1})")
        print(f"==============================\n")

        model = YOLO(args.model)

        # determine how many layers we actually have
        try:
            total_layers = len(list(model.model.model))
        except Exception:
            total_layers = len(list(model.model.children()))

        n_clamped = max(0, min(int(n), total_layers))
        freeze_indices = list(range(n_clamped))  # 0..n-1 frozen

        # set requires_grad flags
        set_frozen_first_n_layers(model, n_clamped)
        # keep frozen layers in eval mode
        set_eval_for_frozen_layers(model, freeze_indices)

        # ensure they stay in eval mode during training
        def _make_eval_cb(indices):
            def _cb(trainer):
                try:
                    set_eval_for_frozen_layers(trainer.model, indices)
                except Exception:
                    pass
            return _cb

        model.add_callback('on_train_start', _make_eval_cb(freeze_indices))
        model.add_callback('on_train_epoch_start', _make_eval_cb(freeze_indices))

        print(f"[freeze_experiment] total_layers={total_layers} "
              f"frozen_layers={freeze_indices}")
        summarize_trainable_and_bn(model, f'freeze_1_to_{n_clamped}')

        exp_name = f'freeze_1_to_{n_clamped}'
        model.train(
            data=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            single_cls=False,
            batch=args.batch,
            close_mosaic=args.close_mosaic,
            workers=args.workers,
            device=args.device,
            optimizer=args.optimizer,
            amp=args.amp,
            project=args.project,
            name=exp_name,
            freeze=freeze_indices,
        )

    os.system('shutdown')
