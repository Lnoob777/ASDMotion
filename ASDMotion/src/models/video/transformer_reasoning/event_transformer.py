import torch
from torch import nn
class VideoTransformer(nn.Module):
    def __init__(self,T=256,Heads=8,scalars_dim=8,num_encode = 3,num_decode = 0,dim_ff = 2048,dropout=0.5,active="gelu",batch_first=True,norm_first=True,bias=True,num_event_types=32,event_type_emb_dim=32):
        super().__init__()
        self.d_model = T
        self.pos_embedding = nn.Parameter(torch.zeros(1,65,T))
        self.type_emb = nn.Embedding(num_event_types,event_type_emb_dim)
        self.token_fuse = nn.Sequential(nn.Linear(T+scalars_dim+1+event_type_emb_dim,T),nn.GELU(),nn.LayerNorm(T))

        self.input_proj = nn.Linear(T+event_type_emb_dim,T)
        self.cls_head = nn.Sequential(nn.Linear(T,T//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(T//2,1))
        self.conf_head = nn.Sequential(nn.Linear(T,T//4),nn.GELU(),nn.Dropout(dropout),nn.Linear(T//4,1))
        self.transformer = nn.Transformer(d_model=T,nhead=Heads,num_encoder_layers=num_encode,num_decoder_layers=num_decode,dim_feedforward=dim_ff,dropout=dropout,activation=active,batch_first=batch_first,norm_first=norm_first,bias=bias)
        
        
        self.norm = nn.LayerNorm(T)
    def forward(self,x):
        tokens = x["tokens"]
        attn_mask = x["attn_mask"]
        event_type_id = x["event_type_id"]
        token_conf = x["token_conf"]
        event_scalars = x["event_scalars"]


        B,K,D = tokens.shape
        assert D==self.d_model

        token_conf = token_conf.unsqueeze(-1)
        event_type_emb = self.type_emb(event_type_id)
        X= torch.cat([tokens,event_scalars,token_conf,event_type_emb],dim=-1)
        X=self.token_fuse(X)


        X= X+self.pos_embedding[:,:X.size(1),:]
        src_key_padding_mask = ~attn_mask
        X = self.transformer.encoder(X,src_key_padding_mask=src_key_padding_mask)
        X = self.norm(X)
        valid = attn_mask.float()              # [B, K]
        valid = valid.unsqueeze(-1)            # [B, K, 1]

        X = X * valid                          # zero out padded tokens
        z = X.sum(dim=1) / (valid.sum(dim=1).clamp(min=1.0))   # [B, 256]

        logit = self.cls_head(z).squeeze(-1)
        p = torch.sigmoid(logit)
        conf_logit = self.conf_head(z).squeeze(-1)
        conf_p = torch.sigmoid(conf_logit)
        return {
            "z":z,
            "logit":logit,
            "prob":p,
            "confidence score":conf_p
        }
        

