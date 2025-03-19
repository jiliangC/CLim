import torch

from vim.models_mamba import vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=True).to(device=device)

inputs = torch.randn(1, 3, 224, 224).to(device)
out = model(inputs)
print(out.shape)