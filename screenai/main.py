
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn
from torch.autograd import Function
from zeta.nn import (
    SwiGLU,
    FeedForward,
    Attention,
)
from zeta.structs import (
    Encoder,
    ViTransformerWrapper,
)

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def divisible_by(numer, denom):
    return (numer % denom) == 0


# distributed


def pad_dim_to(t, length, dim=0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))


def all_gather_variable_batch(t):
    device, rank, world_size = (
        t.device,
        dist.get_rank(),
        dist.get_world_size(),
    )

    size = torch.tensor(t.shape[0], device=device, dtype=torch.long)
    sizes = [
        torch.empty_like(size, device=device, dtype=torch.long)
        for i in range(world_size)
    ]
    dist.all_gather(sizes, size)

    sizes = torch.stack(sizes)
    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim=0)
    gathered_tensors = [
        torch.empty_like(
            padded_t, device=device, dtype=padded_t.dtype
        )
        for i in range(world_size)
    ]
    dist.all_gather(gathered_tensors, padded_t)

    gathered_tensor = torch.cat(gathered_tensors)
    seq = torch.arange(max_size, device=device)

    mask = rearrange(seq, "j -> 1 j") < rearrange(sizes, "i -> i 1")
    mask = rearrange(mask, "i j -> (i j)")

    gathered_tensor = gathered_tensor[mask]
    sizes = sizes.tolist()

    return gathered_tensor, sizes


class AllGather(Function):
    @staticmethod
    def forward(ctx, x):
        assert dist.is_initialized() and dist.get_world_size() > 1
        x, batch_sizes = all_gather_variable_batch(x)
        ctx.batch_sizes = batch_sizes
        return x

    @staticmethod
    def backward(ctx, grads):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim=0)
        return grads_by_rank[rank]


all_gather = AllGather.apply


# normalization
# they use layernorm without bias, something that pytorch does not offer


# to latents


class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)


# parallel attention and feedforward with residual
# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.context_norm = (
            nn.LayerNorm(context_dim)
            if norm_context
            else nn.Identity()
        )

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = (
            nn.Sequential(
                nn.Linear(dim, ff_inner_dim * 2, bias=False),
                SwiGLU(),
                nn.Linear(ff_inner_dim, dim, bias=False),
            )
            if parallel_ff
            else None
        )

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge and combine heads

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)

        return out


class MultiModalEncoder(nn.Module):
    """
    MultiModalEncoder class is responsible for encoding multi-modal inputs using self-attention mechanism.

    Args:
        dim (int): The dimension of the input and output tensors. Default is 512.
        depth (int): The number of layers in the encoder. Default is 6.
        dim_head (int): The dimension of each head in the self-attention mechanism. Default is 64.
        heads (int): The number of attention heads. Default is 8.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The dimension of the input and output tensors.
        depth (int): The number of layers in the encoder.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each head in the self-attention mechanism.
        layers (list): List of attention and feedforward layers.

    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        # self.layers = nn.ModuleList([])

        # for _ in range(depth):
        #     attention_layer = Attention(dim, dim_head, heads, causal=True, qk_norm=True, flash="cuda")
        #     feedforward_layer = FeedForward(dim, dim, 4, *args, **kwargs)

        #     self.layers.append(attention_layer)
        #     self.layers.append(feedforward_layer)

        self.layers = nn.ModuleList(
            [Attention(dim, dim_head, heads, causal=True, qk_norm=True, flash="cuda"),
            FeedForward(dim, dim, 4, *args, **kwargs)]
            for _ in range(depth)
        )
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MultiModalEncoder.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The encoded tensor.

        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


class MultiModalDecoder(nn.Module):
    """
    MultiModalDecoder module for decoding multi-modal inputs.

    Args:
        dim (int): The dimension of the input.
        depth (int): The number of layers in the decoder.
        dim_head (int): The dimension of each attention head.
        heads (int): The number of attention heads.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The dimension of the input.
        depth (int): The number of layers in the decoder.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        layers (nn.ModuleList): List of decoder layers.

    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        *args,
        **kwargs,
    ):
        # super(self, MultiModalDecoder).__init__(*args, **kwargs)
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head

        self.layers = nn.ModuleList(
            [
                (
                    CrossAttention(
                        dim,
                        context_dim=dim,
                        heads=heads,
                        *args,
                        **kwargs,
                    ),
                    Attention(
                        dim,
                        dim_head,
                        heads,
                        causal=True,
                        qk_norm=True,
                        flash="cuda",
                    ),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for cross_attn, attn in self.layers:
            x = cross_attn(x, x) + x
            x = attn(x) + x

        return x


class ScreenAI(nn.Module):
    """
    ScreenAI module for multimodal learning.

    Args:
        patch_size (int): Size of the image patches.
        image_size (int): Size of the input image.
        dim (int): Dimension of the model.
        depth (int): Depth of the model.
        dim_head (int): Dimension of the attention head.
        heads (int): Number of attention heads.
        vit_depth (int): Depth of the ViT transformer.
        multi_modal_encoder_depth (int): Depth of the multimodal encoder.
        llm_decoder_depth (int): Depth of the LLM decoder.
        mm_encoder_ff_mult (int): Multiplier for the feed-forward dimension in the multimodal encoder.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        patch_size (int): Size of the image patches.
        image_size (int): Size of the input image.
        dim (int): Dimension of the model.
        depth (int): Depth of the model.
        heads (int): Number of attention heads.
        vit_depth (int): Depth of the ViT transformer.
        multi_modal_encoder_depth (int): Depth of the multimodal encoder.
        llm_decoder_depth (int): Depth of the LLM decoder.
        patch_embedding (nn.Conv2d): Patch embedding layer.
        vit (ViTransformerWrapper): ViT transformer layer.
        image_embedding (nn.Linear): Image embedding layer.
        to_out (nn.Sequential): Output layer.
        flash (str): Device to use for computation.
        encoder (MultiModalEncoder): Multimodal encoder layer.
        decoder (MultiModalDecoder): LLM decoder layer.
    """

    def __init__(
        self,
        patch_size: int,
        image_size: int = 224,
        dim: int = 512,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        vit_depth: int = 4,
        multi_modal_encoder_depth: int = 4,
        llm_decoder_depth: int = 4,
        mm_encoder_ff_mult: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.vit_depth = vit_depth
        self.multi_modal_encoder_depth = multi_modal_encoder_depth
        self.llm_decoder_depth = llm_decoder_depth



        # ViTransformerWrapper
        self.vit = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            post_emb_norm=True,
            attn_layers=Encoder(
                dim=dim, depth=vit_depth, heads=heads
            ),
        )

        # Image embedding
        self.image_embedding = nn.Linear(dim, dim)

        # To out
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Softmax(dim=-1)
        )

        # If cuda is avaialble then cuda
        self.flash = "cuda" if torch.cuda.is_available() else "cpu"

        # MultiModal Encoder layers
        self.encoder = MultiModalEncoder(
            dim,
            multi_modal_encoder_depth,
            dim_head,
            heads,
        )

        # LLM Layer / T5
        self.decoder = MultiModalDecoder(
            dim,
            llm_decoder_depth,
            dim_head,
            heads,
        )

    def forward(self, text: Tensor, img: Tensor) -> Tensor:
        """
        Forward pass of the ScreenAI module.

        Args:
            text (Tensor): Input text tensor.
            img (Tensor): Input image tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Image patch
        img = rearrange(
            img,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        print(f"Image patch shape: {img.shape}")

        # vit
        img = self.vit(img, return_embeddings=True)

        # Embed image
        img = self.image_embedding(img)

        # Concatenate image and text
        x = torch.cat((img, text), dim=1)
        print(x.shape)

        # T5 Multimodal encoder
        x = self.encoder(x)

        # Pass the k, v values into the cross attention of llm
        x = self.decoder(x)

        # To out
        x = self.to_out(x)

        return x
