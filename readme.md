**Status:** Waiting...

# ExplainRegex

Code and models from the paper "ExplainRegex: Automatic Description Generation for Regular Expressions via Neural Machine Translation".

The code relies heavily on [transformers](https://github.com/huggingface/transformers) , [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers) and [QKNorm](https://github.com/CyndxAI/QKNorm). 

## How to change Self-Attention to Norm-Attention

Self-Attention:

```python
proj_shape = (bsz * self.num_heads, -1, self.head_dim)
query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
key_states = key_states.view(*proj_shape)
value_states = value_states.view(*proj_shape)

src_len = key_states.size(1)
attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
```

Norm-Attention:

```python
proj_shape = (bsz * self.num_heads, -1, self.head_dim)
query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
key_states = key_states.view(*proj_shape)
value_states = value_states.view(*proj_shape)

query_states = F.normalize(query_states, p=2, dim=-1)
key_states = F.normalize(key_states, p=2, dim=-1)
scaleup = self.scaleup
src_len = key_states.size(1)
attn_weights = scaleup(torch.matmul(query_states, key_states.transpose(1, 2)))
```

where scaleup is:

```python
self.scaleup = ScaleUp(scaling_factor(sequence_threshold))
## init sequence_threshold=30
```

```python
def scaling_factor(sequence_threshold):
    return np.log2((sequence_threshold ** 2) - sequence_threshold)

class ScaleUp(nn.Module):
    """ScaleUp"""

    def __init__(self, scale):
        super(ScaleUp, self).__init__()
        self.scale = Parameter(torch.tensor(scale))

    def forward(self, x):
        return x * self.scale
```

## How to train?

Replace modeling_bart.py in transformers with modeling_bart.py in this project and just run train.py