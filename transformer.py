import torch
import torch.nn.functional as F

torch.set_default_device("cuda")
BATCH_SIZE = 1
SEQ_SIZE = 4
EMBEDDING_SIZE = 2
ATTENTION_HEADS = 4


class SelfAttention(torch.nn.Module):
    def __init__(self, k=EMBEDDING_SIZE, heads=ATTENTION_HEADS, mask=False) -> None:
        super().__init__()
        assert (
            k % heads == 0
        ), "Número de dimensões do embedding não divisível pelo número de cabeças de atenção"
        self.k, self.heads = k, heads
        self.chunksize = self.k // self.heads
        self.tokeys = torch.nn.Linear(in_features=k, out_features=k, bias=False)
        self.toqueries = torch.nn.Linear(in_features=k, out_features=k, bias=False)
        self.tovalues = torch.nn.Linear(in_features=k, out_features=k, bias=False)
        self.unifyheads = torch.nn.Linear(in_features=k, out_features=k)

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

        # calculating self attention
        dot = F.softmax(dot, dim=2)
        # encoding self attention step
        outs = torch.bmm(dot, kqv_list[2]).view(
            batch_size, self.heads, seq_size, self.chunksize
        )
        outs = outs.transpose(1, 2).contiguous().view(batch_size, seq_size, self.k)
        return self.unifyheads(outs)
