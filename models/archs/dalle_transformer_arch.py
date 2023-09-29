import math
from functools import wraps

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

from models.archs.einops_exts import repeat_many
from models.archs.rotary_embedding_torch import RotaryEmbedding


# helper functions
def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, dim=-1)


# relative positional bias for causal transformer
class RelPosBias(nn.Module):

    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  num_buckets=32,
                                  max_distance=128):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) /
                                    math.log(max_distance / max_exact) *
                                    (num_buckets - max_exact)).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large)

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-5, fp16_eps=1e-3, stable=False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


# attention
class Attention(nn.Module):

    def __init__(self,
                 dim,
                 *,
                 dim_head=64,
                 heads=8,
                 dropout=0.,
                 causal=False,
                 rotary_emb=None,
                 cosine_sim=True,
                 cosine_sim_scale=16):
        super().__init__()
        self.scale = cosine_sim_scale if cosine_sim else (dim_head**-0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim))

    def forward(self, x, mask=None, attn_bias=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale

        # rotary embeddings

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b 1 d', b=b)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # whether to use cosine sim

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool,
                                     device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = self.dropout(attn)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# feedforward
class SwiGLU(nn.Module):
    """ used successfully in https://arxiv.org/abs/2204.0231 """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


def FeedForward(dim, mult=4, dropout=0., post_activation_norm=False):
    """ post-activation norm https://arxiv.org/abs/2110.09456 """

    inner_dim = int(mult * dim)
    return nn.Sequential(
        LayerNorm(dim), nn.Linear(dim, inner_dim * 2, bias=False), SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout), nn.Linear(inner_dim, dim, bias=False))


class NonCausalTransformerLanguage(nn.Module):

    def __init__(self,
                 *,
                 dim,
                 depth,
                 dim_head=64,
                 heads=8,
                 ff_mult=4,
                 norm_in=False,
                 norm_out=True,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 final_proj=True,
                 normformer=False,
                 rotary_emb=True):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity(
        )  # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads=heads)

        self.text_feature_mapping = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.Linear(256, dim),
            nn.LayerNorm(dim),
        )

        rotary_emb = RotaryEmbedding(
            dim=min(32, dim_head)) if rotary_emb else None

        self.mask_emb = nn.Parameter(torch.zeros(1, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(
                        dim=dim,
                        causal=False,
                        dim_head=dim_head,
                        heads=heads,
                        dropout=attn_dropout,
                        rotary_emb=rotary_emb),
                    FeedForward(
                        dim=dim,
                        mult=ff_mult,
                        dropout=ff_dropout,
                        post_activation_norm=normformer)
                ]))

        self.norm = LayerNorm(
            dim, stable=True
        ) if norm_out else nn.Identity(
        )  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(
            dim, dim, bias=False) if final_proj else nn.Identity()

    def forward(self, x, exemplar_frame_embeddings, text_embedding, masks):
        x_masked = x.clone()
        x_masked[masks, :] = self.mask_emb

        x = torch.cat((self.text_feature_mapping(text_embedding),
                       exemplar_frame_embeddings, x_masked),
                      dim=1).clone()

        n, device = x.shape[1], x.device
        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device=device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)