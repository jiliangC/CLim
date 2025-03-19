import torch
from torch import nn

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# Residual CLIP Adapter
class MasterAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(MasterAdapter, self).__init__()
        self.fc1 = nn.Linear(c_in, bottleneck, bias=False)
        self.ln1 = nn.LayerNorm(bottleneck)
        self.leaky_relu1 = nn.LeakyReLU(inplace=False)
        self.fc2 = nn.Linear(bottleneck, c_in, bias=False)
        self.ln2 = nn.LayerNorm(c_in)
        self.leaky_relu2 = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.fc1(x)  # [16 * 197, 768]
        x = self.ln1(x)
        x = self.leaky_relu1(x)

        y = self.fc2(x)  # [16*197, 192]
        y = self.ln2(y)
        y = self.leaky_relu2(y)

        return x, y


class SlaverAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(SlaverAdapter, self).__init__()
        self.fc1 = nn.Linear(c_in, bottleneck, bias=False)
        self.leaky_relu1 = nn.LeakyReLU(inplace=False)
        self.fc2 = nn.Linear(bottleneck, c_in, bias=False)
        self.leaky_relu2 = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        y = self.fc2(x)
        y = self.leaky_relu2(y)
        return x, y


class VIM_Inplanted(nn.Module):
    def __init__(self, vim, features):
        super().__init__()
        self.vim = vim

        self.features = features
        self.master_adapters = nn.ModuleList([MasterAdapter(384, bottleneck=768) for i in range(len(features))])
        self.slaver_adapters = nn.ModuleList([SlaverAdapter(384, bottleneck=768) for i in range(len(features))])

    def forward(self, x):
        global slaver_adapt_out, slaver_adapt_med
        x = self.vim.patch_embed(x)  # x [1,196,192]
        B, M, _ = x.shape
        cls_token = self.vim.cls_token.expand(B, -1, -1)
        token_position = M // 2
        # add cls token in the middle
        x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)  # x [1,197,192]
        x = x + self.vim.pos_embed
        x = self.vim.pos_drop(x)

        residual = None
        hidden_states = x

        master_patch_tokens = []
        slaver_patch_tokens = []
        for i, layer in enumerate(self.vim.layers):
            hidden_states, residual = layer(hidden_states, residual)

            if i+2 in self.features:
                slaver_adapt_med, slaver_adapt_out = self.slaver_adapters[self.features.index(i + 2)](hidden_states)

            if i+1 in self.features:
                master_adapt_med, master_adapt_out = self.master_adapters[self.features.index(i + 1)](hidden_states)  # seg_adapt_med [290,16,768] seg_adapt_out[290,16,1024]
                # slaver_adapt_med, slaver_adapt_out = self.slaver_adapters[self.features.index(i + 1)](hidden_states)
                hidden_states = 0.8 * x + 0.1 * master_adapt_out + 0.1 * slaver_adapt_out
                master_patch_tokens.append(master_adapt_med)
                slaver_patch_tokens.append(slaver_adapt_med)

        # remove cls
        master_patch_tokens = [torch.cat((master_patch_tokens[t][:, :token_position, :], master_patch_tokens[t][:, token_position+1:, :]), dim=1) for t in range(len(master_patch_tokens))]
        slaver_patch_tokens = [torch.cat((slaver_patch_tokens[t][:, :token_position, :], slaver_patch_tokens[t][:, token_position+1:, :]), dim=1) for t in range(len(slaver_patch_tokens))]

        return master_patch_tokens,slaver_patch_tokens
