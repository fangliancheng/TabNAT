import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

randn_like=torch.randn_like
SIGMA_MIN=0.002
SIGMA_MAX=80
rho=7
S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1

class Mixed_Loss(nn.Module):
    def __init__(self, n_num, categories, hid_dim, dim_t = 1024, dropout_rate = 0.3):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(categories)

        self.hid_dim = hid_dim
        self.num_predictor = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 16),
            nn.ReLU(),
            nn.Linear(hid_dim * 16, n_num)
        )

        self.cat_preditors = nn.ModuleList()

        # Two layer MLP for each category
        for n_cat_i in categories:
            self.cat_preditors.append(nn.Sequential(
                nn.Linear(hid_dim, hid_dim * 4),
                nn.ReLU(),
                nn.Linear(hid_dim * 4, hid_dim * 16),
                nn.ReLU(),
                nn.Linear(hid_dim * 16, hid_dim * 4),
                nn.ReLU(),
                nn.Linear(hid_dim * 4, n_cat_i)
            ))


        self.num_loss = DiffLoss(d_in = n_num, dim_t = dim_t, dropout_rate=dropout_rate)
        self.cat_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, z_num, z_cat, gt_num, gt_cat, mask):
        losses = []
        if z_num is not None:      
            num_loss = self.num_loss(gt_num, self.num_predictor(z_num)).mean(1).unsqueeze(1)
            losses.append(num_loss)
        if z_cat is not None:
            for i, cat_predictor in enumerate(self.cat_preditors):
                pred_cat_i = cat_predictor(z_cat[:,i])
                # print(np.sort(pred_cat_i.softmax(dim=-1)[0].cpu().detach().numpy()))
                loss_i = self.cat_loss(pred_cat_i, gt_cat[:,i]).unsqueeze(1)

                losses.append(loss_i)

        losses = torch.cat(losses, dim=1)
        loss = (losses * mask).sum() / mask.sum()   
        
        # if (1-mask).sum() > 0:
        #     loss = loss + (losses*(1-mask)).sum() / (1-mask).sum()
                
        if self.n_num > 0:
            loss_num = losses[:, 0]
            mask_num = mask[:, 0]
            loss_num = (loss_num * mask_num).sum() / mask_num.sum()
        else:
            loss_num = 0

        if self.n_cat > 0:
            loss_cat = losses[:, -self.n_cat:]
            mask_cat = mask[:, -self.n_cat:]
            loss_cat = (loss_cat * mask_cat).sum() / mask_cat.sum()
        else:
            loss_cat = 0


        return loss, loss_num, loss_cat

    def sample(self, z_num, z_cat, num_steps = 50, device = 'cuda'):
        B = z_num.shape[0] if z_num is not None else z_cat.shape[0]

        if z_num is not None:
            z_pred = self.num_predictor(z_num)
            sampled_num = self.num_loss.sample(B, self.n_num, z_pred, num_steps, device)
        else:
            sampled_num = None

        if z_cat is not None:
            sampled_cat = []
            for i, cat_predictor in enumerate(self.cat_preditors):
                pred_cat_i = cat_predictor(z_cat[:,i])
                probs_i = F.softmax(pred_cat_i / 1.0, dim=-1)
                sample_cat_i = torch.multinomial(probs_i, num_samples=1)
                sampled_cat.append(sample_cat_i)
                
            sampled_cat = torch.cat(sampled_cat, dim=1)
        else:
            sampled_cat = None

        return sampled_num, sampled_cat

    def cond_sample(self, miss_x_num, miss_mask_num, z_num, z_cat, num_steps = 50, device = 'cuda'):
        B = z_num.shape[0] if z_num is not None else z_cat.shape[0]

        if z_num is not None:
            z_pred = self.num_predictor(z_num)
            sampled_num = self.num_loss.impute_mask(miss_x_num, miss_mask_num, B, self.n_num, z_pred, num_steps, device)
        else:
            sampled_num = None

        if z_cat is not None:
            sampled_cat = []
            for i, cat_predictor in enumerate(self.cat_preditors):
                pred_cat_i = cat_predictor(z_cat[:,i])
                probs_i = F.softmax(pred_cat_i / 1.0, dim=-1)

                # Apply top-k sampling
                # k = min(2, probs_i.shape[-1])  # Choose k, e.g., 3 or less if fewer categories
                # top_k_probs = top_k_sampling(probs_i, k)
                # sample_cat_i = torch.multinomial(top_k_probs, num_samples=1)
    
                sample_cat_i = torch.multinomial(probs_i, num_samples=1)
                sampled_cat.append(sample_cat_i)
                
            sampled_cat = torch.cat(sampled_cat, dim=1)
        else:
            sampled_cat = None

        return sampled_num, sampled_cat

def top_k_sampling(probs, k):
    top_k = torch.topk(probs, k)
    indices_to_remove = probs < top_k.values[:, -1].unsqueeze(-1)
    probs[indices_to_remove] = 0
    return probs / probs.sum(dim=-1, keepdim=True)


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class MLPDiffusion_dropout(nn.Module):
    def __init__(self, d_in, dim_t=512, dropout_rate=0.1):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_t, dim_t)
        )
        self.z_embed = nn.Sequential(
            nn.Linear(d_in, dim_t),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, noise_labels, z):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)

        emb += self.z_embed(z)
    
        x = self.proj(x) + emb
        return self.mlp(x)

class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t = 512):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        self.z_embed = nn.Sequential(
            nn.Linear(d_in, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, noise_labels, z):

        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)

        emb += self.z_embed(z)
        
        x = self.proj(x) + emb
        return self.mlp(x)

class DiffLoss(nn.Module):
    "Conditional Diffusion Loss"
    def __init__(self, d_in, dim_t,
        P_mean = -1.2,
        P_std = 1.2,
        sigma_data = 0.5,
        gamma = 5,
        sigma_min = 0,
        sigma_max = float('inf'),
        dropout_rate = 0.3):
        super().__init__()

        self.denoise_fn = MLPDiffusion(d_in, dim_t)
        #self.denoise_fn = MLPDiffusion_dropout(d_in, dim_t, dropout_rate=dropout_rate)
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.gamma = gamma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def precond(self, denoise_fn, x, z, sigma):

        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        x_in = c_in * x
        F_x = denoise_fn(x_in, c_noise.flatten(), z)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x


    def forward(self, data, z):

        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = self.precond(self.denoise_fn, y + n, z, sigma)
    
        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)

        return loss

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def sample_step(self, z, num_steps, i, t_cur, t_next, x_next):
        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = self.round_sigma(t_cur + gamma * t_cur) 
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # Euler step.

        denoised = self.precond(self.denoise_fn, x_hat, z,  t_hat.expand(x_next.shape[0], 1)).to(torch.float32)
        d_cur = (x_hat - denoised) / t_hat
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.

        if i < num_steps - 1:
            denoised = self.precond(self.denoise_fn, x_next, z, t_next.expand(x_next.shape[0], 1)).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sample(self, B, embed_dim, z, num_steps = 50, device = 'cuda'):

        latents = torch.randn([B, embed_dim], device=device)
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

        sigma_min = max(SIGMA_MIN, self.sigma_min)
        sigma_max = min(SIGMA_MAX, self.sigma_max)

        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_next = latents.to(torch.float32) * t_steps[0]

        with torch.no_grad():
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                x_next = self.sample_step(z, num_steps, i, t_cur, t_next, x_next)

        return x_next
    
    def impute_mask(self, x, mask, B, embed_dim, z, num_steps = 50, device = 'cuda:0'):
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        x_t = torch.randn([B, embed_dim], device=device)

        sigma_min = max(SIGMA_MIN, self.sigma_min)
        sigma_max = min(SIGMA_MAX, self.sigma_max)

        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        mask = mask.to(torch.int).to(device)
        x_t = x.to(torch.float32) * t_steps[0]

        N = 10
        with torch.no_grad():

            for i, (t_cur, t_next) in tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])), total=num_steps-1):
                if i < num_steps - 1:
            
                    for j in range(N):
                        n_curr = torch.randn_like(x_t).to(device) * t_cur
                        n_prev = torch.randn_like(x_t).to(device) * t_next

                        x_known_t_prev = x + n_prev
                        x_unknown_t_prev = self.sample_step(z, num_steps, i, t_cur, t_next, x_t)

                        x_t_prev = x_known_t_prev * (1-mask) + x_unknown_t_prev * mask

                        n = torch.randn_like(x_t) * (t_cur.pow(2) - t_next.pow(2)).sqrt()

                        if j == N - 1:
                            x_t = x_t_prev                                                # turn to x_{t-1}
                        else:
                            x_t = x_t_prev + n                                            # new x_t

        return x_t



