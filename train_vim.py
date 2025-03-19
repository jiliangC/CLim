import warnings
from doctest import master
import time
import yaml
import os


from vim.vim.models_mamba import vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
from vim.vim.vim_adapter import VIM_Inplanted

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


from easydict import EasyDict

warnings.filterwarnings("ignore")

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
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME



os.environ["TOKENIZERS_PARALLELISM"] = "false"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


CLASS_INDEX = {'BUSI':4,'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main(args):
    with open(args.config_path) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    setup_seed(args.config.seed)

    clip_model = create_model(model_name=args.config.model_name, img_size=args.config.img_size, device=device, pretrained=args.config.pretrain, require_pretrained=True)
    clip_model.eval()

    vim = vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=True).to(device=device)
    vim.eval()
    model = VIM_Inplanted(vim, args.config.features_list).to(device)
    model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = True



    # optimizer for only adapters
    master_optimizer = torch.optim.Adam(list(model.master_adapters.parameters()), lr=args.config.learning_rate, betas=(0.5, 0.999))
    slaver_optimizer = torch.optim.Adam(list(model.slaver_adapters.parameters()), lr=args.config.learning_rate, betas=(0.5, 0.999))


    # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.config.data_path, args.obj, args.config.img_size, args.shot, args.config.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)


    # few-shot image augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)

    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.config.batch_size, shuffle=True, **kwargs)


    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    # ablation prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    best_result = 0

    for epoch in range(args.config.epoch):
        print('epoch ', epoch+1, ':')

        loss_list = []

        for (image, gt, label) in tqdm(train_loader):
            image = image.to(device)
            with torch.cuda.amp.autocast():
                master_patch_tokens, slaver_patch_tokens = model(image)

                # det loss
                slaver_loss = 0
                image_label = label.to(device)
                for layer in range(len(slaver_patch_tokens)):
                    slaver_patch_tokens[layer] = slaver_patch_tokens[layer] / slaver_patch_tokens[layer].norm(dim=-1,keepdim=True)
                    anomaly_map = (100.0 * slaver_patch_tokens[layer] @ text_features)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    slaver_loss += loss_bce(anomaly_score, image_label)

                # pixel level
                master_loss = 0
                if CLASS_INDEX[args.obj] > 0:
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(master_patch_tokens)):
                        master_patch_tokens[layer] = master_patch_tokens[layer] / master_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * master_patch_tokens[layer] @ text_features)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.config.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        master_loss += loss_focal(anomaly_map, mask)
                        master_loss += loss_dice(anomaly_map[:, 1, :, :], mask)
                else:
                    for layer in range(len(master_patch_tokens)):
                        master_patch_tokens[layer] = master_patch_tokens[layer] / master_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * master_patch_tokens[layer] @ text_features)
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        anomaly_score = torch.mean(anomaly_map, dim=-1)
                        master_loss += loss_bce(anomaly_score, image_label)

                loss = master_loss + slaver_loss
                loss.requires_grad_(True)

                master_optimizer.zero_grad()
                slaver_optimizer.zero_grad()
                loss.backward()
                master_optimizer.step()
                slaver_optimizer.step()


                loss_list.append(loss.item())

        print("Loss: {}".format(np.mean(loss_list)))

        master_features = []
        for image in tqdm(support_loader):
            image = image[0].to(device)
            with torch.no_grad():
                master_patch_tokens, slaver_patch_tokens = model(image)
                master_patch_tokens = [p[0].contiguous() for p in master_patch_tokens]
                master_features.append(master_patch_tokens)
        master_mem_features = [torch.cat([master_features[j][i] for j in range(len(master_features))], dim=0) for i in range(len(master_features[0]))]


        result = test(args, model, test_loader, text_features, master_mem_features)
        if result > best_result:
            best_result = result
            print("Best result\n")
            if args.config.save_model == 1:
                ckp_path = os.path.join(args.config.save_path, f'{args.obj}.pth')
                torch.save({'master_adapters': model.master_adapters.state_dict(),
                            'slaver_adapters': model.slaver_adapters.state_dict(),},
                            ckp_path)

def test(args, model, test_loader, text_features, master_mem_features):
    start_time = time.time()
    current_memory = 0
    max_memory_after = 0

    gt_list = []
    gt_mask_list = []

    master_score_map_first = []
    master_score_map_second= []

    for (image, label, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            master_patch_tokens,_ = model(image)
            master_patch_tokens = [p[0, :, :] for p in master_patch_tokens]

            if CLASS_INDEX[args.obj] > 0:


                anomaly_maps_few_shot = []
                for idx, p in enumerate(master_patch_tokens):
                    cos = cos_sim(master_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(anomaly_map_few_shot.clone().detach(),
                                                            size=args.config.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                master_score_map_second.append(score_map_few)


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
                master_score_map_first.append(score_map_zero)
            else:
                anomaly_maps_few_shot = []
                for idx, p in enumerate(master_patch_tokens):
                    cos = cos_sim(master_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                         size=args.config.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                score_few_det = anomaly_map_few_shot.mean()
                master_score_map_second.append(score_few_det)


                anomaly_score = 0
                for layer in range(len(master_patch_tokens)):
                    master_patch_tokens[layer] /= master_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * master_patch_tokens[layer] @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                master_score_map_first.append(anomaly_score.cpu().numpy())

            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(label.cpu().detach().numpy())
        # 打印当前的显存使用情况
        current_memory = max(torch.cuda.memory_allocated(),current_memory)
        max_memory_after = max(torch.cuda.max_memory_allocated(),max_memory_after)
    print(f"当前已分配内存: {current_memory / 1024 / 1024 / 1024:.2f} GB")
    print(f"最大显存使用（峰值）: {max_memory_after / 1024 / 1024 / 1024:.2f} GB")


    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)

    master_score_map_first = np.array(master_score_map_first)
    master_score_map_second = np.array(master_score_map_second)

    master_score_map_first = (master_score_map_first - master_score_map_first.min()) / (
                master_score_map_first.max() - master_score_map_first.min())
    master_score_map_second = (master_score_map_second - master_score_map_second.min()) / (
                master_score_map_second.max() - master_score_map_second.min())

    if CLASS_INDEX[args.obj] > 0:
        segment_scores = 0.5 * master_score_map_first + 0.5 * master_score_map_second
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')
        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        img_roc_auc_det = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(img_roc_auc_det, 4)}')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Test time per epoch: {elapsed_time:.2f} seconds')
        return seg_roc_auc + img_roc_auc_det
    else:
        segment_scores = 0.5 * master_score_map_first + 0.5 * master_score_map_second
        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        img_roc_auc_det = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(img_roc_auc_det, 4)}')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Test time per epoch: {elapsed_time:.2f} seconds')
        return img_roc_auc_det


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--config_path", type=str, default='config.yaml', help="model configs")
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument("--shot", type=int, default=4, help="image number")
    args = parser.parse_args()
    main(args)



