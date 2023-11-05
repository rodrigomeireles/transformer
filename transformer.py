import torch
import torch.nn.functional as F
from torch import nn

torch.set_default_device("cuda")
BATCH_SIZE = 1
SEQ_SIZE = 4
EMBEDDING_SIZE = 4
ATTENTION_HEADS = 2


class SelfAttention(nn.Module):
    def __init__(self, k=EMBEDDING_SIZE, heads=ATTENTION_HEADS, mask=False) -> None:
        super().__init__()
        assert (
            k % heads == 0
        ), "Número de dimensões do embedding não divisível pelo número de cabeças de atenção"
        self.k, self.heads = k, heads
        self.chunksize = self.k // self.heads
        self.tokeys = nn.Linear(in_features=k, out_features=k, bias=False)
        self.toqueries = nn.Linear(in_features=k, out_features=k, bias=False)
        self.tovalues = nn.Linear(in_features=k, out_features=k, bias=False)
        self.unifyheads = nn.Linear(in_features=k, out_features=k)
        self.mask = mask

    # Ref: https://peterbloem.nl/blog/transformers
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, seq_size, token_size = input.size()
        assert token_size == self.k, "Dimensão de input errada."
        kqv_list = []
        for func in self.tokeys, self.toqueries, self.tovalues:
            output = func(input).view(batch_size, seq_size, self.heads, self.chunksize)
            kqv_list.append(
                output.transpose(1, 2)
                .contiguous()
                .view(batch_size * self.heads, seq_size, self.chunksize)
            )

        dot = torch.bmm(kqv_list[1], kqv_list[0].transpose(1, 2)) / (self.k ** (1 / 2))
        # mask for autoregression
        if self.mask:
            indices = torch.triu_indices(seq_size, seq_size, offset=1)
            dot[:, *indices] = float("-inf")

        # calculating self attention
        dot = F.softmax(dot, dim=2)
        # encoding self attention step
        outs = torch.bmm(dot, kqv_list[2]).view(
            batch_size, self.heads, seq_size, self.chunksize
        )
        outs = outs.transpose(1, 2).contiguous().view(batch_size, seq_size, self.k)
        return self.unifyheads(outs)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads) -> None:
        super().__init__()

        self.attention = SelfAttention(k=k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        # 4 vezes maior porque o tutorial quis
        self.ff = nn.Sequential(nn.Linear(k, 4 * k), nn.ReLU(), nn.Linear(4 * k, k))
        self.norm2 = nn.LayerNorm(k)

    def forward(self, x: torch.Tensor):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        return self.norm2(fedforward + x)


class ClassifierTransformer(nn.Module):
    def __init__(
        self,
        k: int,
        heads: int,
        depth: int,
        seq_size: int,
        input_size: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.token_emb = nn.Embedding(num_embeddings=input_size, embedding_dim=k)
        self.pos_emb = nn.Embedding(num_embeddings=seq_size, embedding_dim=k)
        self.tblocks = nn.Sequential(
            *[TransformerBlock(k=k, heads=heads) for i in range(depth)]
        )
        self.almost_probs = nn.Linear(k, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        tokens = self.token_emb(x)
        b, t, k = tokens.size()
        positions = self.pos_emb(torch.arange(t))[None, :, :].expand(b, t, k)
        x = self.tblocks(x + positions)
        x = self.almost_probs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)
