import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from tabnat.loss import Mixed_Loss
from tqdm import tqdm

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


class TabNAT(nn.Module):
    def __init__(self, n_num, n_cat, categories, embed_dim=32, buffer_size=8, depth=6, norm_layer=nn.LayerNorm, dropout_rate=0.0, device='cuda:0'):
        super().__init__()
        self.n_num = n_num
        self.n_cat = n_cat
        self.categories = categories
        self.buffer_size = buffer_size
        self.embed_dim = embed_dim
        self.device = device

        self.seq_len = int(n_num != 0) + n_cat

        print(f'n_num: {n_num}, n_cat: {n_cat}, seq_len: {self.seq_len}')

        num_heads = 4
        mlp_ratio = 16.0
        self.label_drop_prob = 0.1

        # Conditional generation
        self.class_embedding = nn.Embedding(categories[0], embed_dim)

        # Unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, embed_dim))
        self.tokenizer = Tokenizer(n_num, categories, embed_dim)

        # Position embeddings
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + buffer_size, embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + buffer_size, embed_dim))
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))

        # Transformer encoder
        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.encoder_norm = norm_layer(embed_dim)

        # Transformer decoder
        self.decoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.decoder_norm = norm_layer(embed_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Loss
        self.mixed_loss = Mixed_Loss(n_num, categories, embed_dim, dim_t=1024, dropout_rate=dropout_rate)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.class_embedding.weight, std=0.02)
        torch.nn.init.normal_(self.fake_latent, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=0.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def sample_orders(self, B):
        orders = [torch.randperm(self.seq_len) for _ in range(B)]
        return torch.stack(orders).to(self.device)
    
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

    def random_masking(self, x, orders):
        B, seq_len, _ = x.shape
        mask_num = np.random.randint(1, seq_len + 1)
        mask = torch.zeros(B, seq_len, device=self.device)
        return torch.scatter(mask, dim=-1, index=orders[:, :mask_num], src=torch.ones_like(mask))

    def forward_mae_encoding(self, x, mask, class_embedding):
        B, _, D = x.shape

        if self.buffer_size > 0:
            x = torch.cat([torch.zeros(B, self.buffer_size, D, device=self.device), x], dim=1)
            mask = torch.cat([torch.zeros(B, self.buffer_size, device=self.device), mask], dim=1)
        
        # random drop class embedding during training
        if self.training and class_embedding is not None:
            drop_latent_mask = torch.rand(B) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
        
        x[:, :self.buffer_size] = class_embedding.unsqueeze(1) if class_embedding is not None else self.fake_latent

        x = x + self.encoder_pos_embed_learned

        x_after_pad = torch.zeros_like(x)
        x_after_pad[(1 - mask).bool()] = x[(1 - mask).bool()]
        x = x_after_pad

        for block in self.encoder_blocks:
            x = block(x)

        return self.encoder_norm(x)

    def forward_mae_decoding(self, x, mask):
        B, _, D = x.shape

        mask = torch.cat([torch.zeros(B, self.buffer_size, device=self.device), mask], dim=1)

        mask_tokens = self.mask_token.expand(B, mask.shape[1], D)
        x_padded = mask_tokens.clone()
        x_padded[(1 - mask).bool()] = x[(1 - mask).bool()]

        x = x_padded + self.decoder_pos_embed_learned

        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)
        x = x[:, self.buffer_size:] + self.diffusion_pos_embed_learned
        return x

    def forward(self, x_num, x_cat, cls=None):
        class_emb = self.class_embedding(cls.squeeze()) if cls is not None else None

        gt_num = x_num.clone().detach() if x_num is not None else None
        gt_cat = x_cat.clone().detach() if x_cat is not None else None

        x_num, x_cat = self.tokenizer(x_num, x_cat)

        x = torch.cat([v for v in [x_num, x_cat] if v is not None], dim=1)

        orders = self.sample_orders(x.size(0))
        mask = self.random_masking(x, orders)
        
        x_enc = self.forward_mae_encoding(x, mask, class_emb)
        z = self.forward_mae_decoding(x_enc, mask)

        if x_num is not None and x_cat is not None:
            z_num, z_cat = z[:, 0], z[:, 1:]
        elif x_num is not None:
            z_num, z_cat = z.squeeze(1), None
        else:
            z_num, z_cat = None, z

        loss, loss_num, loss_cat = self.mixed_loss(z_num=z_num, z_cat=z_cat, gt_num=gt_num, gt_cat=gt_cat, mask=mask)
        return loss, loss_num, loss_cat

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
    
    