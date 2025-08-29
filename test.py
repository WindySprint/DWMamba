import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.optim
from tqdm import tqdm
from collections import OrderedDict

from models import network
from utils import dataloader
from utils import dataloader_Sobel, dataloader_Scharr, dataloader_Laplacian

torch.cuda.empty_cache()
parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--net_name', type=str, default="net_U")
parser.add_argument('--dataset_name', type=str, default="UIEB")
parser.add_argument('--ori_images_path', type=str, default="../../../share/UIE/datasets/test")

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
parser.add_argument('--result_path', type=str, default="./results/")
parser.add_argument('--cudaid', type=str, default="0", help="choose cuda device id 0-7).")

config = parser.parse_args()
print("gpu_id:", config.cudaid)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def test(config):
    dims = [24, 48, 96, 48, 24]
    depths = [1, 1, 2, 1, 1]
    # dims = [48, 96, 192, 96, 48]
    # depths = [2, 2, 9, 2, 2]
    enhan_net = network.DWMamba(dims=dims, depths=depths).cuda()

    load_checkpoint(enhan_net, os.path.join(config.checkpoint_path, config.net_name, 'model_best.pth'))

    if len(config.cudaid) > 1:
        cudaid_list = config.cudaid.split(",")
        cudaid_list = [int(x) for x in cudaid_list]
        device_ids = [i for i in cudaid_list]
        enhan_net = nn.DataParallel(enhan_net, device_ids=device_ids)
        
    print(os.path.join(config.ori_images_path, config.dataset_name))
    # test_dataset = dataloader.test_loader(os.path.join(config.ori_images_path, config.dataset_name))
    test_dataset = dataloader.test_loader(os.path.join(config.ori_images_path, config.dataset_name))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                               num_workers=config.num_workers, drop_last=False, pin_memory=True)
    # create results folds
    if not os.path.exists(os.path.join(config.result_path, config.net_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name))

    if not os.path.exists(os.path.join(config.result_path, config.net_name, config.dataset_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name, config.dataset_name))

    result_dir = os.path.join(config.result_path, config.net_name, config.dataset_name)

    enhan_net.eval()

    with torch.no_grad():
        for _, (img_ori, img_canny, filenames) in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img_ori = img_ori.cuda()
            img_canny = img_canny.cuda()
            enhan_image = enhan_net(img_ori, img_canny)
            for i in range(len(enhan_image)):                
                torchvision.utils.save_image(enhan_image[i], os.path.join(result_dir, filenames[i]))
    from thop import profile
    flops, params = profile(enhan_net, inputs=(img_ori,img_canny))
    print('flops: %.4f G, params: %.4f M' % (flops, params))

if __name__ == '__main__':
    start_time = time.time()
    test(config)
    print("test_time:"+str((time.time()-start_time)))
