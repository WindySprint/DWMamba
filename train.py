import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

from models import network
from utils import dataloader, losses

torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
# Input Parameters
parser.add_argument('--net_name', type=str, default="net_U")
parser.add_argument('--enhan_images_path', type=str, default="../../../share/UIE/datasets/train/UIEB/gt/")
parser.add_argument('--ori_images_path', type=str, default="../../../share/UIE/datasets/train/UIEB/input/")
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--train_batch_size', type=int, default=2)
parser.add_argument('--val_batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--checkpoint_path', type=str, default="checkpoints/")
parser.add_argument('--cudaid', type=str, default="1",help="choose cuda device id).")
config = parser.parse_args()

def train(config):
    print("gpu_id:", config.cudaid)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid
    
    dims = [24, 48, 96, 48, 24]
    depths = [1, 1, 2, 1, 1]
    # dims = [48, 96, 192, 96, 48]
    # depths = [2, 2, 9, 2, 2]
    enhan_net = network.DWMamba(dims=dims, depths=depths).cuda()

    if not os.path.exists(os.path.join(config.checkpoint_path, config.net_name)):
        os.mkdir(os.path.join(config.checkpoint_path, config.net_name))

    if len(config.cudaid) > 1:
        cudaid_list = config.cudaid.split(",")
        cudaid_list = [int(x) for x in cudaid_list]
        device_ids = [i for i in cudaid_list]
        enhan_net = nn.DataParallel(enhan_net, device_ids=device_ids)

    criterion_char = losses.Charbonnier_Loss()
    criterion_ssim = losses.SSIM_Loss()
    # criterion_vl = losses.VL_Loss()
    # criterion_sl1 = nn.SmoothL1Loss()
    # criterion_vgg = losses.VGG_Loss()

    train_dataset = dataloader.train_val_loader(config.enhan_images_path,config.ori_images_path)
    val_dataset = dataloader.train_val_loader(config.enhan_images_path,config.ori_images_path, mode="val")
    # train_dataset = dataloader_Scharr.train_val_loader(config.enhan_images_path, config.ori_images_path)
    # val_dataset = dataloader_Scharr.train_val_loader(config.enhan_images_path, config.ori_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,num_workers=config.num_workers, pin_memory=True)

    ######### Adam optimizer ###########
    optimizer = optim.Adam(enhan_net.parameters(), lr=config.lr)
    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs - warmup_epochs,
                                                            eta_min=config.lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    enhan_net.train()

    # Record best index and corresponding epoch
    best_psnr = 0
    best_epoch = 0

    for epoch in range(1, config.num_epochs+1):
        epoch_start_time = time.time()
        # Record train loss and validation index
        train_loss = []
        val_psnr = []
        print("*" * 30 + "The %i epoch" % epoch + "*" * 30+'\n')
        
        for _, (img_clean, img_ori, img_canny) in enumerate(tqdm(train_loader)):
            img_clean = img_clean.cuda()
            img_ori = img_ori.cuda()
            img_canny = img_canny.cuda()

            try:
                enhanced_image = enhan_net(img_ori, img_canny)
                char_loss = criterion_char(img_clean, enhanced_image)
                ssim_loss = criterion_ssim(img_clean, enhanced_image)
                # vl_loss = criterion_vl(img_clean, enhanced_image)
                ssim_loss = 1 - ssim_loss
                sum_loss = char_loss + 0.5 * ssim_loss

                # sum_loss = char_loss + 0.5 * ssim_loss + 2 * vl_loss

                # sl1_loss = criterion_sl1(img_clean, enhanced_image)
                # sum_loss = sl1_loss + 0.05 * ssim_loss
                # vgg_loss = criterion_vgg(img_clean, enhanced_image)
                # sum_loss = char_loss + 10 * sl1_loss + 0.5 * ssim_loss
                # sum_loss = sl1_loss + 0.01 * vgg_loss

                train_loss.append(sum_loss.item())
                optimizer.zero_grad()
                sum_loss.backward()
                torch.nn.utils.clip_grad_norm_(enhan_net.parameters(), config.grad_clip_norm)
                optimizer.step()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e

        with open(os.path.join(config.checkpoint_path, config.net_name, "loss.log"), "a+", encoding="utf-8") as f:
            s = "The %i Epoch mean_loss is :%f" % (epoch, np.mean(train_loss)) + "\n"
            f.write(s)

        # Validation Stage
        with torch.no_grad():
            for _, (img_clean, img_ori, img_canny) in enumerate(val_loader):
                img_clean = img_clean.cuda()
                img_ori = img_ori.cuda()
                img_canny = img_canny.cuda()
                enhanced_image = enhan_net(img_ori, img_canny)

                psnr = losses.torchPSNR(img_clean, enhanced_image)
                val_psnr.append(psnr.item())

        val_psnr = np.mean(np.array(val_psnr))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            torch.save({'state_dict': enhan_net.state_dict()},
                       os.path.join(config.checkpoint_path, config.net_name, "model_best.pth"))
        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format
              (epoch, time.time() - epoch_start_time, np.mean(train_loss), scheduler.get_lr()[0]))

        print("------------------------------------------------------------------")

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
              (epoch, val_psnr, best_epoch, best_psnr))

        with open(os.path.join(config.checkpoint_path, config.net_name, "val_PSNR.log"), "a+", encoding="utf-8") as f:
            f.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
                    (epoch, val_psnr, best_epoch, best_psnr) + "\n")

        torch.save({'state_dict': enhan_net.state_dict()},
                   os.path.join(config.checkpoint_path, config.net_name, "model_latest.pth"))

if __name__ == "__main__":
    start_time = time.time()
    train(config)
    e = time.time()
    print("train_time:"+str(time.time()-start_time))

