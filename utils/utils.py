import logging
import numpy as np
import torch
import os


def m_print(log):
    logging.info(log)
    print(log)
    return


def save_checkpoints(epoch, path, is_best=False, model=None, optimizer=None):
    state_dict = {
        "epoch": epoch,
        "model": model.cpu().state_dict(),
        "optimizer": optimizer.state_dict()
    }
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(state_dict, os.path.join(path, "latest_model.tar"))
    torch.save(state_dict["model"], os.path.join(path, f"model_{str(epoch).zfill(4)}.pth"))
    if is_best:
        print(f'{model.__class__.__name__},Found best score in {epoch} epoch')
        torch.save(state_dict, os.path.join(path, "best_model.tar"))
    model.cuda()
    return model


def load_checkpoints(path, model):
    pth_path = os.path.join(path, "best_model.tar")
    # pth_path = os.path.join(path, "model_0005.pth")
    model_checkpoint = torch.load(pth_path)
    model_static_dict = model_checkpoint["model"]
    # model_static_dict = model_checkpoint
    checkpoint_epoch = model_checkpoint['epoch']
    model.optimizer = model_checkpoint['optimizer']
    # checkpoint_epoch = 5
    print(f"Loading {model.__class__.__name__} checkpoint, epoch = {checkpoint_epoch}")
    model.load_state_dict(model_static_dict)
    model.cuda()
    model.eval()
    return model
