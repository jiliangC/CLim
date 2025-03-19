import warnings
from doctest import master

import yaml
import os

from vim.vim.models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2, \
    vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2, \
    vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2, \
    vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
from vim.vim.vim_adapter import VIM_Inplanted

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


from easydict import EasyDict

warnings.filterwarnings("ignore")
from torch.utils.data import Subset
import os
import argparse
import random
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from dataset.medical_few import MedDataset
from clip.clip import create_model

from sklearn.metrics import roc_auc_score

from utils import augment, cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME



os.environ["TOKENIZERS_PARALLELISM"] = "false"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1 }



def main(args):
    with open(args.config_path) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    clip_model = create_model(model_name=args.config.model_name, img_size=args.config.img_size, device=device, pretrained=args.config.pretrain, require_pretrained=True)
    clip_model.eval()

    vim = vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=True).to(device=device)
    vim.eval()
    model = VIM_Inplanted(vim, args.config.features_list).to(device)
    model.eval()
    checkpoint = torch.load(os.path.join(f'{args.config.save_path}', f'{args.obj}.pth'))
    model.master_adapters.load_state_dict(checkpoint["master_adapters"])
    model.slaver_adapters.load_state_dict(checkpoint["slaver_adapters"])

    for name, param in model.named_parameters():
        param.requires_grad = False

    # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.config.data_path, args.obj, args.config.img_size, args.shot, args.config.iterate)
    # Select the first 10 samples from the test dataset
    # indices = list(range(1))  # Select the first 10 samples
    # test_subset = Subset(test_dataset, indices)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # few-shot image augmentation
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)



    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)


    master_features = []
    for image in tqdm(support_loader):
        image = image[0].to(device)
        with torch.no_grad():
            master_patch_tokens, slaver_patch_tokens = model(image)
            master_patch_tokens = [p[0].contiguous() for p in master_patch_tokens]
            master_features.append(master_patch_tokens)
    master_mem_features = [torch.cat([master_features[j][i] for j in range(len(master_features))], dim=0) for i in range(len(master_features[0]))]
    test(args, model, test_loader, text_features, master_mem_features)



def test(args, model, test_loader, text_features, master_mem_features):
    gt_list = []
    gt_mask_list = []

    master_score_map_zero = []
    master_score_map_few= []
    index = 1
    for (image, label, mask) in tqdm(test_loader):
        label_value = label.item()  # 获取label的标量值

        if label_value == 0:  # 如果label为tensor(0)
            continue
        image = image.to(device)
        # mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            master_patch_tokens,_ = model(image)
            master_patch_tokens = [p[0, :, :] for p in master_patch_tokens]

            anomaly_maps_few_shot = []
            for idx, p in enumerate(master_patch_tokens):
                cos = cos_sim(master_mem_features[idx], p)
                height = int(np.sqrt(cos.shape[1]))
                anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                anomaly_map_few_shot = F.interpolate(anomaly_map_few_shot.clone().detach(),
                                                        size=args.config.img_size, mode='bilinear', align_corners=True)
                anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
            score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
            master_score_map_few.append(score_map_few)

            anomaly_maps = []
            for layer in range(len(master_patch_tokens)):
                master_patch_tokens[layer] /= master_patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * master_patch_tokens[layer] @ text_features).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.config.img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            score_map_zero = np.sum(anomaly_maps, axis=0)
            master_score_map_zero.append(score_map_zero)


            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(label.cpu().detach().numpy())


        gt_list = np.array(gt_list)
        gt_mask_list = np.asarray(gt_mask_list)
        gt_mask_list = (gt_mask_list>0).astype(np.int_)

        master_score_map_zero = np.array(master_score_map_zero)
        master_score_map_few = np.array(master_score_map_few)

        master_score_map_zero = (master_score_map_zero - master_score_map_zero.min()) / (
                    master_score_map_zero.max() - master_score_map_zero.min())
        master_score_map_few = (master_score_map_few - master_score_map_few.min()) / (
                    master_score_map_few.max() - master_score_map_few.min())


        segment_scores = 0.5 * master_score_map_zero + 0.5 * master_score_map_few #shape[1,1,224,224]
        # gt_mask_list.shape #shape[1,224,224]

        display_images(segment_scores, gt_mask_list,image,index,'image_res/'+args.obj)
        index += 1


        gt_list = []
        gt_mask_list = []
        master_score_map_zero = []
        master_score_map_few = []


import numpy as np
import matplotlib.pyplot as plt


def display_images(segment_scores, gt_mask_list,image, index, save_dir='image_res'):
    """
    将 segment_scores 和 gt_mask_list 显示为图像
    """
    # 检查保存路径是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 处理 segment_scores
    score_map = segment_scores[0]  # shape: (224, 224)
    score_map = score_map.squeeze()

    # 处理 gt_mask_list
    mask = gt_mask_list[0]  # shape: (224, 224)
    mask = mask.astype(np.uint8)  # 转换为 0 或 1

    # 显示图像
    plt.figure(figsize=(30, 8))

    # 原图
    # 处理 image，去掉批次维度并调整通道顺序
    image_ = image.squeeze().permute(1, 2, 0).cpu().numpy()  # shape: (224, 224, 3)
    plt.subplot(1, 4, 1)  # 第一个子图
    plt.imshow(image_)
    plt.title(f"Original Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)  # 第二个子图
    plt.imshow(score_map, cmap='jet')
    plt.title(f"Segment Score Map")
    plt.axis('off')

    score_map[score_map > 0.5], score_map[score_map <= 0.5] = 1, 0
    plt.subplot(1, 4, 3)  # 第三个子图
    plt.imshow(score_map, cmap='gray')
    plt.title(f"Segment Score Map2")
    plt.axis('off')

    plt.subplot(1, 4, 4)  # 第三个子图
    plt.imshow(mask, cmap='gray')
    plt.title(f"Ground Truth Mask")
    plt.axis('off')

    # 保存图像到文件夹
    save_path = os.path.join(save_dir, f"image_{index}.png")  # 按照索引保存
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # 关闭当前图像


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--config_path", type=str, default='config.yaml', help="model configs")
    parser.add_argument('--obj', type=str, default='Retina_RESC')
    parser.add_argument("--shot", type=int, default=16, help="image number")
    args = parser.parse_args()
    main(args)



