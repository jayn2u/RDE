import os
import os.path as op
import argparse

import torch

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RDE evaluation")
    parser.add_argument("--checkpoint_dir", required=True, type=str,
                        help="training output dir containing configs.yaml and best.pth/last.pth")
    parser.add_argument("--config_file", default=None, type=str,
                        help="path to configs.yaml (default: <checkpoint_dir>/configs.yaml)")
    parser.add_argument("--root_dir", default=None, type=str,
                        help="override dataset root (default: value saved in configs.yaml)")
    parser.add_argument("--gpu", default=None, type=str,
                        help="CUDA device id(s), e.g. 0 or 0,1")
    cli_args = parser.parse_args()

    if cli_args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cli_args.gpu

    config_file = cli_args.config_file or op.join(cli_args.checkpoint_dir, 'configs.yaml')
    if not op.exists(config_file):
        raise FileNotFoundError(f"config file not found: {config_file}")

    args = load_train_configs(config_file)
    args.training = False
    args.output_dir = cli_args.checkpoint_dir
    if cli_args.root_dir is not None:
        args.root_dir = cli_args.root_dir

    logger = setup_logger('RDE', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")

    device = torch.device("cuda")
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    checkpoints = ['best.pth', 'last.pth']
    for ckpt_name in checkpoints:
        ckpt_path = op.join(args.output_dir, ckpt_name)
        if op.exists(ckpt_path):
            model = build_model(args, num_classes)
            checkpointer = Checkpointer(model)
            checkpointer.load(f=ckpt_path)
            model = model.to(device)
            do_inference(model, test_img_loader, test_txt_loader)
