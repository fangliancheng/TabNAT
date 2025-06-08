import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from timm.models.vision_transformer import Attention, Mlp
from baselines.dp_tbart.models.loss import Mixed_Loss

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.5) # Liancheng: enable dropout
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class Tokenizer(nn.Module):
    def __init__(self, n_num, categories, embed_dim):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(categories)
        self.categories = categories
        self.embed_dim = embed_dim

        self.num_embeddings = SimpleMLP(n_num, embed_dim, embed_dim)
        self.cat_embeddings = nn.ModuleList()
        
        for i in range(self.n_cat):
            self.cat_embeddings.append(nn.Embedding(self.categories[i], embed_dim))

    def forward(self, x_num, x_cat):
        if x_num is not None:
            x_num = self.num_embeddings(x_num).unsqueeze(1)
        if x_cat is not None:
            x_cats = []
            for i in range(self.n_cat):
                x_cat_i = self.cat_embeddings[i](x_cat[:,i]).unsqueeze(1)
                x_cats.append(x_cat_i)
            x_cats = torch.cat(x_cats, dim = 1)
        else:
            x_cats = None

        return x_num, x_cats


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self, 
        in_dim, 
        hid_dim, 
        out_dim,
        num_res_blocks=4
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_res_blocks = num_res_blocks

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return 



class TabMAR(nn.Module):
    def __init__(self, n_num, n_cat, categories, embed_dim=32, buffer_size=8, depth=3, norm_layer=nn.LayerNorm, dropout_rate=0.0, padding=False, device='cuda:0'):
        super().__init__()

        # ------------------------------------------------
        self.n_num = n_num
        self.categories = categories
        #self.n_cat = len(categories) 
        self.n_cat = n_cat
        self.buffer_size = buffer_size
        self.embed_dim = embed_dim
        self.device = device
        self.attn_drop = dropout_rate
        self.proj_drop = dropout_rate
        self.padding = padding

        self.seq_len = 0
        if n_num != 0:
            self.seq_len += 1
        if self.n_cat != 0:
            self.seq_len += self.n_cat

        print(f'n_num: {n_num}, n_cat: {self.n_cat}, seq_len: {self.seq_len}')

        num_heads = 4
        mlp_ratio = 16.0

        encoder_depth = depth
        decoder_depth = depth

        # Class embedding for conditional generation
        self.class_embedding = nn.Embedding(self.categories[0], embed_dim) 
        self.label_drop_prob = 0.1
        # Class embedding for unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, embed_dim))

        # MAR tokenizer:
        self.tokenizer = Tokenizer(n_num, categories, embed_dim)

        # MAR encoder specifics
    
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, embed_dim))
        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads = num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
            proj_drop=self.proj_drop, attn_drop=self.attn_drop) for _ in range(encoder_depth)
        ])
        
        # self.encoder_blocks = nn.ModuleList([
        #     DiTBlock(embed_dim, num_heads = 4, mlp_ratio=4., attn_drop=0.0) for _ in range(encoder_depth)
        # ])
    
        self.encoder_norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

        # MAR decoder specifics
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads = num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
            proj_drop=self.proj_drop, attn_drop=self.attn_drop) for _ in range(decoder_depth)
        ])
        # self.decoder_blocks = nn.ModuleList([
        #     DiTBlock(embed_dim, num_heads = 4, mlp_ratio=4.0, attn_drop=0.0) for _ in range(decoder_depth)
        # ])

        self.decoder_norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))

        self.mixed_loss = Mixed_Loss(self.n_num, categories, embed_dim, dim_t = 1024, dropout_rate = self.attn_drop)
        self.initialize_weights()

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_embedding.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def sample_orders(self, B):
        # generate a batch of random generation orders
        orders = []
        for _ in range(B):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)

        orders = torch.LongTensor(orders).to(self.device)

        return orders

    def sample_subset_orders(self, B, masks):
        # given masks, generate the rest of the tokens randomly
        orders = []
        for i in range(B):
            cur_mask = masks[i]
            cur_idx = cur_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            order = list(range(self.seq_len))
            order_rest = np.array(list(set(order) - set(cur_idx)))
            np.random.shuffle(order_rest)
            order = np.concatenate([cur_idx, order_rest])
            orders.append(order)
        
        orders = torch.LongTensor(orders).to(self.device)
        return orders

    def mask_by_order(self, mask_len, orders, bsz, seq_len, device):
        masking = torch.zeros(bsz, seq_len).to(device)
        if isinstance(mask_len, int):
            masking = torch.scatter(masking, dim=-1, index=orders[:, :mask_len], src=torch.ones(bsz, seq_len).to(device)).bool()
        elif isinstance(mask_len, torch.Tensor):
            for i in range(bsz):
                masking[i] = torch.scatter(masking[i], dim=-1, index=orders[i, :mask_len[i]], src=torch.ones(seq_len, device=device)).bool()
        else:
            raise ValueError(f'mask_len should be int or torch.Tensor, but got {type(mask_len)}')
        return masking

    def random_masking(self, x, orders):
        # generate token mask
        B, seq_len, embed_dim = x.shape
        mask_num = np.random.randint(1, seq_len+1)
        mask = torch.zeros(B, seq_len, device=self.device)
        mask = torch.scatter(mask, dim = -1, index = orders[:,:mask_num], src = torch.ones(B, seq_len, device=self.device))

        return mask

    def forward_mae_encoding(self, x, mask, class_embedding):
        B, seq_len, embed_dim = x.shape

        # concat buffer
        if self.buffer_size > 0:    
            x = torch.cat([torch.zeros(B, self.buffer_size, embed_dim, device = self.device), x], dim = 1)
            mask_with_buffer = torch.cat([torch.zeros(B, self.buffer_size, device = self.device), mask], dim = 1)
        
        # random drop class embedding during training
        if self.training and class_embedding is not None:
            drop_latent_mask = torch.rand(B) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
        
        if class_embedding is None:
            x[:, :self.buffer_size] = self.fake_latent #TODO: check if autocast properly
        else:
            x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        
        # encoder pos embedding
        x = x + self.encoder_pos_embed_learned

        if not self.padding:    
            x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(B, -1, embed_dim)
        else:
            # pad zero or mask_token to x so that it can be reshaped
            x_after_pad = torch.zeros(B, mask_with_buffer.shape[1], embed_dim, device=self.device)
            #mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
            #x_after_pad = mask_tokens.clone()
            x_after_pad[(1-mask_with_buffer).nonzero(as_tuple=True)] = x[(1-mask_with_buffer).nonzero(as_tuple=True)]
            x = x_after_pad
        
        # apply Transformer blocks
        for block in self.encoder_blocks:
            x = block(x)

        x = self.encoder_norm(x)

        return x    

    def forward_mae_decoding(self, x, mask):

        mask_with_buffer = torch.cat([torch.zeros(x.shape[0], self.buffer_size, device=self.device), mask], dim = 1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        if not self.padding:
            x_after_pad[(1-mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        else:
            x_after_pad[(1-mask_with_buffer).nonzero(as_tuple=True)] = x[(1-mask_with_buffer).nonzero(as_tuple=True)]

        # decoder pos embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned

        return x


    def forward(self, x_num, x_cat, cls=None):
        if cls is not None:
            cls = cls.squeeze()
            class_emb = self.class_embedding(cls)
        else:
            class_emb = None

        if x_num is not None:
            gt_num = x_num.clone().detach()
        else:
            gt_num = None
        if x_cat is not None:
            gt_cat = x_cat.clone().detach()
        else:
            gt_cat = None
   
        x_num, x_cat = self.tokenizer(x_num, x_cat) 

        if x_num is not None and x_cat is not None:
            x = torch.cat([x_num, x_cat], dim=1)
        elif x_num is not None:
            x = x_num
        elif x_cat is not None:
            x = x_cat

        orders = self.sample_orders(x.shape[0])
        mask = self.random_masking(x, orders)

        # MAE encoding
        x = self.forward_mae_encoding(x, mask, class_emb)

        # MAE decoding
        z = self.forward_mae_decoding(x, mask)

        if x_num is not None and x_cat is not None:
            z_num = z[:, 0]
            z_cat = z[:, 1:]
        elif x_num is not None:
            z_num = z.squeeze(1)
            z_cat = None
        elif x_cat is not None:
            z_num = None
            z_cat = z

        loss = self.mixed_loss(z_num = z_num, z_cat = z_cat, gt_num = gt_num, gt_cat = gt_cat, mask = mask)

        return loss

    def sample(self, bsz, cls=None, device = 'cuda:0'):

        syn_num = torch.zeros(bsz, self.n_num).to(device) if self.n_num > 0 else None
        syn_cat = torch.zeros(bsz, self.n_cat).long().to(device) if self.n_cat > 0 else None   

        # init and sample generation orders
        mask_len = self.seq_len
        mask = torch.ones(bsz, self.seq_len).to(device)
        tokens = torch.zeros(bsz, self.seq_len, self.embed_dim).to(device)
        orders = self.sample_orders(bsz)

        indices = list(range(self.seq_len))

        for step in tqdm(indices):
            cur_tokens = tokens.clone()
            
            if cls is not None:
                cls = cls.squeeze()
                class_embedding = self.class_embedding(cls) 
            else:
                class_embedding = None

            x = self.forward_mae_encoding(tokens, mask, class_embedding)
            
            z = self.forward_mae_decoding(x, mask)

            if self.n_num > 0 and self.n_cat > 0:
                z_num = z[:, 0]
                z_cat = z[:, 1:]
            elif self.n_num > 0:
                z_num = z.squeeze(1)
                z_cat = None
            elif self.n_cat > 0:
                z_num = None
                z_cat = z

            mask_len = mask_len - 1

            # get masking for next iteration and locations to be predicted in this iteration
            # predict next token one by one
            mask_next = self.mask_by_order(mask_len, orders, bsz, self.seq_len, device)
            
            if step >= self.seq_len - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())

            mask = mask_next

            if self.n_num > 0:
                mask_to_pred_num = mask_to_pred[:, 0]
            else:
                mask_to_pred_num = None
            if self.n_cat > 0:
                mask_to_pred_cat = mask_to_pred[:, -self.n_cat:]
            else:
                mask_to_pred_cat = None

            sampled_num, sampled_cat = self.mixed_loss.sample(z_num = z_num, z_cat = z_cat, num_steps = 50, device = self.device)

            if self.n_num > 0:
                syn_num[mask_to_pred_num.nonzero(as_tuple=True)] = sampled_num[mask_to_pred_num.nonzero(as_tuple=True)]
            if self.n_cat > 0:
                syn_cat[mask_to_pred_cat.nonzero(as_tuple=True)] =  sampled_cat[mask_to_pred_cat.nonzero(as_tuple=True)]
            
            if step < self.seq_len - 1:
                sampled_num_tokens, sampled_cat_tokens = self.tokenizer(sampled_num, sampled_cat)
            
                if sampled_num_tokens is not None and sampled_cat_tokens is not None:
                    sampled_tokens = torch.cat([sampled_num_tokens, sampled_cat_tokens], dim = 1)
                elif sampled_num_tokens is not None:
                    sampled_tokens = sampled_num_tokens
                elif sampled_cat_tokens is not None:
                    sampled_tokens = sampled_cat_tokens

                cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_tokens[mask_to_pred.nonzero(as_tuple=True)]
                tokens = cur_tokens.clone()

        return syn_num, syn_cat
    
    def impute(self, miss_x_num, miss_x_cat, miss_mask, cls=None, one_step=False, device = 'cuda:0'):
        miss_mask = miss_mask.to(device)
        miss_mask_np = miss_mask.cpu().numpy()
        miss_mask_num_np = miss_mask_np[:, :self.n_num]
        
        if miss_x_num is not None:
            bsz = miss_x_num.shape[0]
            miss_x_num = miss_x_num.to(device)
            miss_x_num_np = miss_x_num.cpu().numpy()
            miss_x_num_np[miss_mask_num_np == 1] = np.nan
            mean_imp_x_num = np.nan_to_num(miss_x_num_np, nan=np.nanmean(miss_x_num_np, axis=0)) 
            mean_imp_x_num = torch.tensor(mean_imp_x_num).to(device)
        else:
            mean_imp_x_num = None

        if miss_x_cat is not None:
            bsz = miss_x_cat.shape[0]
            miss_x_cat = miss_x_cat.to(device)

        miss_mask_num = miss_mask[:, :self.n_num]
        miss_mask_cat = miss_mask[:, self.n_num:]

        syn_num = torch.zeros(bsz, self.n_num).to(device) if self.n_num > 0 else None
        syn_cat = torch.zeros(bsz, self.n_cat).long().to(device) if self.n_cat > 0 else None   

        # construct mask from miss_mask
        mask = torch.ones(bsz, self.seq_len).to(device)
        
        mask_num_idx = ~torch.any(miss_mask_num.bool(), dim=1) # num part
        mask[mask_num_idx, 0] = 0
        
        if self.n_cat > 0:
            mask[:, -self.n_cat:] = miss_mask_cat  # cat part

        mask_len = mask.sum(dim=1).long()  # different init mask_len for different samples
        
        init_num_tokens, init_cat_tokens = self.tokenizer(x_num=mean_imp_x_num, x_cat=miss_x_cat)
        
        # or, simply use miss_x_num 
        #init_num_tokens, init_cat_tokens = self.tokenizer(x_num=miss_x_num, x_cat=miss_x_cat) # mean_imp_x_num /approx miss_x_num due to normalization

        # miss_mask_cat: bsz * n_cat, init_cat_tokens: bsz * n_cat * embed_dim.
        # if miss_mask_cat[i,j] == 1, then set init_cat_tokens[i,j,:] = 0
        if miss_x_cat is not None:
            init_cat_tokens[miss_mask_cat.bool().unsqueeze(-1).expand_as(init_cat_tokens)] = 0 

        if init_num_tokens is not None and init_cat_tokens is not None:
            tokens_init = torch.cat([init_num_tokens, init_cat_tokens], dim = 1)
        elif init_num_tokens is not None:
            tokens_init = init_num_tokens
        elif init_cat_tokens is not None:
            tokens_init = init_cat_tokens
        
        tokens = tokens_init.clone()

        orders = self.sample_subset_orders(bsz, mask)

        indices = list(range(self.seq_len)) 
       
        if one_step:
            indices = range(1)

        for step in indices: 
            print(f'step: {step}')
            cur_tokens = tokens.clone()

            if cls is not None: 
                cls = cls.squeeze()
                class_embedding = self.class_embedding(cls) 
            else:
                class_embedding = None
    
            x = self.forward_mae_encoding(tokens, mask, class_embedding)

            z = self.forward_mae_decoding(x, mask)

            if self.n_num > 0 and self.n_cat > 0:
                z_num = z[:, 0]
                z_cat = z[:, 1:]
            elif self.n_num > 0:
                z_num = z.squeeze(1)
                z_cat = None
            elif self.n_cat > 0:
                z_num = None
                z_cat = z

            mask_len = torch.max(mask_len - 1, torch.zeros_like(mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            # predict next token one by one
            mask_next = self.mask_by_order(mask_len, orders, bsz, self.seq_len, device)
            
            if step >= self.seq_len - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            
            # impute all missing values at once
            if one_step:
                mask_to_pred = mask[:bsz].bool()

            mask = mask_next
            
            if self.n_num > 0:
                mask_to_pred_num = mask_to_pred[:, 0]
            else:
                mask_to_pred_num = None
            if self.n_cat > 0:
                mask_to_pred_cat = mask_to_pred[:, -self.n_cat:]
            else:
                mask_to_pred_cat = None

            sampled_num, sampled_cat = self.mixed_loss.cond_sample(miss_x_num=miss_x_num, 
                                                                   miss_mask_num=miss_mask_num, 
                                                                   z_num = z_num, 
                                                                   z_cat = z_cat, 
                                                                   num_steps = 50,
                                                                   device = self.device)
            
            def replacement(sampled_num, sampled_cat, miss_x_num, miss_x_cat, miss_mask_num, miss_mask_cat):
                sampled_num = sampled_num * miss_mask_num + miss_x_num * (1 - miss_mask_num)
                sampled_cat = sampled_cat * miss_mask_cat + miss_x_cat * (1 - miss_mask_cat)
                return sampled_num, sampled_cat
            
            # optional?
            #sampled_num, sampled_cat = replacement(sampled_num, sampled_cat, miss_x_num, miss_x_cat, miss_mask_num, miss_mask_cat)

            if self.n_num > 0:
                syn_num[mask_to_pred_num.nonzero(as_tuple=True)] = sampled_num[mask_to_pred_num.nonzero(as_tuple=True)]
            if self.n_cat > 0:
                syn_cat[mask_to_pred_cat.nonzero(as_tuple=True)] = sampled_cat[mask_to_pred_cat.nonzero(as_tuple=True)]
            
            if step < self.seq_len - 1:
                sampled_num_tokens, sampled_cat_tokens = self.tokenizer(sampled_num, sampled_cat)
            
                if sampled_num_tokens is not None and sampled_cat_tokens is not None:
                    sampled_tokens = torch.cat([sampled_num_tokens, sampled_cat_tokens], dim = 1)
                elif sampled_num_tokens is not None:
                    sampled_tokens = sampled_num_tokens
                elif sampled_cat_tokens is not None:
                    sampled_tokens = sampled_cat_tokens
                
                cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_tokens[mask_to_pred.nonzero(as_tuple=True)]
                tokens = cur_tokens.clone()

        return syn_num, syn_cat