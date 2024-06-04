import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import onnxruntime
import onnx
from onnx import numpy_helper
from .onnx_export import save_model

class Embd(nn.Module):
    def __init__(self, vocab_size, n_embd, batch, seq):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        # self.wpe = nn.Embedding(block_size, n_embd)
        self.n_embd = n_embd
        self.batch = batch
        self.seq = seq

    def forward(self, idx):
        # device = idx.device
        # b, t = idx.size()

        # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        # pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        pos_emb = torch.ones((1, self.seq, self.n_embd))
        tok_emb = tok_emb + pos_emb

        return tok_emb
    
class LayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.n_embd = n_embd
    
    def forward(self, x):
        x = self.ln(x)
        x = x.view(-1, self.n_embd)
        return x

class RMSNorm(nn.Module):
    def __init__(self, batch, seq, n_embd):
        super().__init__()
        self.eps = 1e-6
        self.scale = nn.Parameter(torch.ones(batch, seq, n_embd))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x = x / rms * self.scale
        return x
    
class Attn(nn.Module):
    def __init__(self, n_head, n_embd, batch, seq):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.batch = batch
        self.seq = seq

    def forward(self, x):
        # B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).view(self.batch, self.seq, -1).split(self.n_embd, dim=2)
        q = q.view(self.batch , self.seq, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = k.view(self.batch, self.seq, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(self.batch, self.seq, self.n_head, self.n_embd // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1))  / (math.sqrt(self.n_embd // self.n_head)) # Remove elem_div_const
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1) # Remove Softmax
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(self.batch * self.seq, self.n_embd) # re-assemble all head outputs side by side

        return y, k, v

def precompute_freqs_cis(dim, end, theta = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    freqs_cis = torch.ones_like(freqs) * freqs + torch.ones_like(freqs) * freqs
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x, n_head, n_embd, seq):
    # ndim = x.ndim
    # assert 0 <= 1 < ndim
    # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    shape = [1, seq, 1, n_embd//n_head//2]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis, n_head, n_embd, seq):
    # xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_ = xq[:,:,:,:n_embd//n_head//2]
    xk_ = xk[:,:,:,:n_embd//n_head//2]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_, n_head, n_embd, seq)
    xq_out = (xq_ * freqs_cis)
    xk_out = (xk_ * freqs_cis)
    xq_out = torch.cat((xq_out, xq_out), dim=-1)
    xk_out = torch.cat((xk_out, xk_out), dim=-1)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RoPE_Attn(nn.Module):
    def __init__(self, n_head, n_embd, batch, seq):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.batch = batch
        self.seq = seq
        self.freqs_cis = precompute_freqs_cis(n_embd // n_head, 2048 * 2)[0:seq]

    def forward(self, x):
        # B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).view(self.batch, self.seq, -1).split(self.n_embd, dim=2)
        q = q.view(self.batch, self.seq, self.n_head, self.n_embd // self.n_head)
        k = k.view(self.batch, self.seq, self.n_head, self.n_embd // self.n_head)
        v = v.view(self.batch, self.seq, self.n_head, self.n_embd // self.n_head)

        q, k = apply_rotary_emb(q, k, self.freqs_cis, self.n_head, self.n_embd, self.seq)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1))  / (math.sqrt(self.n_embd // self.n_head)) # Remove elem_div_const
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1) # Remove Softmax
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(self.batch * self.seq, self.n_embd) # re-assemble all head outputs side by side

        return y, k, v

class QKV_Orca(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.n_embd = n_embd
    
    def forward(self, x):
        x = self.c_attn(x)
        return x

class Attn_Orca(nn.Module):
    def __init__(self, n_head, n_embd, batch, seq):
        super().__init__()
        self.batch = batch
        self.seq = seq
        self.n_embd = n_embd
        self.n_head = n_head

    def forward(self, q, k, v):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = q @ k.transpose(-2, -1)
        att = att / (self.n_embd // self.n_head) # Remove elem_div_const
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1) # Remove Softmax
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(self.batch * self.seq, self.n_embd) # re-assemble all head outputs side by side

        return y

class RoPE_Attn_Orca(nn.Module):
    def __init__(self, n_head, n_embd, batch, seq):
        super().__init__()
        self.batch = batch
        self.seq = seq
        self.n_embd = n_embd
        self.n_head = n_head
        self.freqs_cis = precompute_freqs_cis(n_embd // n_head, 2048 * 2)[0:seq]

    def forward(self, q, k, v):
        v = v.transpose(1, 2)
        # B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k = apply_rotary_emb(q, k, self.freqs_cis, self.n_head, self.n_embd, self.seq)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        att = q @ k.transpose(-2, -1)
        att = att / (self.n_embd // self.n_head) # Remove elem_div_const
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1) # Remove Softmax
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(self.batch * self.seq, self.n_embd) # re-assemble all head outputs side by side

        return y

class Gen(torch.nn.Module):
    def __init__(self, n_head, n_embd, cache_len):
        super().__init__()
        self.cache_len = cache_len
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, q, k, v, cache_K, cache_V):
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        k = torch.cat((k.transpose(-2,-1), cache_K), dim=3)
        v = torch.cat((v, cache_V), dim=2)
        att = torch.matmul(q, k)
        att = torch.div(att, (self.n_embd // self.n_head))
        att = F.softmax(att, dim=-1)
        res = torch.matmul(att, v).transpose(1, 2)
        return res, k, v

class RoPE_Gen(torch.nn.Module):
    def __init__(self, n_head, n_embd, cache_len):
        super().__init__()
        self.cache_len = cache_len
        self.n_head = n_head
        self.n_embd = n_embd
        self.seq = 1
        self.freqs_cis = precompute_freqs_cis(n_embd // n_head, 2048 * 2)[0:self.seq]

    def forward(self, q, k, v, cache_K, cache_V):
        v = v.transpose(1,2)
        q, k = apply_rotary_emb(q, k, self.freqs_cis, self.n_head, self.n_embd, self.seq)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        k = torch.cat((k.transpose(-2,-1), cache_K), dim=3)
        v = torch.cat((v, cache_V), dim=2)
        att = torch.matmul(q, k)
        att = torch.div(att, (self.n_embd // self.n_head))
        att = F.softmax(att, dim=-1)
        res = torch.matmul(att, v).transpose(1, 2)
        return res, k, v
    
class Proj(nn.Module):
    def __init__(self, n_embd, batch, seq):
        super().__init__()
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.batch = batch
        self.seq = seq

    def forward(self, x, y):
        x = y + self.c_proj(x).view(self.batch, self.seq, -1)
        return x
    
class FFN1_with_GELU(nn.Module):
    def __init__(self, n_embd, batch, seq):
        super().__init__()
        # output projection
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.batch = batch
        self.seq = seq

    def forward(self, x):
        x = self.c_fc(x).view(self.batch, self.seq, -1)
        x = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        return x.view(self.batch * self.seq, -1)
    
class FFN2(nn.Module):
    def __init__(self, n_embd, batch, seq):
        super().__init__()
        # output projection
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.batch = batch
        self.seq = seq

    def forward(self, x, y):
        x = y + self.c_proj(x).view(self.batch, self.seq, -1)
        return x

class FFNs_with_SwiGLU(nn.Module):
    def __init__(
        self,
        batch,
        seq,
        dim,
        hidden_dim,
        multiple_of = 256,
        ffn_dim_multiplier = 4,
    ):
        super().__init__()
        self.batch = batch
        self.seq = seq
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)

    def forward(self, x, y):
        w1_output2d = self.w1(x)
        sigmoid2d = F.sigmoid(w1_output2d)
        w1_output3d = w1_output2d.view(self.batch, self.seq, -1)
        sigmoid3d = sigmoid2d.view(self.batch, self.seq, -1)
        w1_output3d = self.w3(x).view(self.batch, self.seq, -1)
        w2_input = ((w1_output3d * sigmoid3d) * w1_output3d).view(self.batch * self.seq, -1)
        x = y + self.w2(w2_input).view(self.batch, self.seq, -1) # residual
        return x

# Sequence
# Embd()
# block
# LayerNorm()
# Attn()
# Proj()
# LayerNorm()
# FFN1()
# FFN2()
# end block
# LayerNorm()
    
def model_export(model_name, config, batch, seq, init_or_gen, half=False, gen_ORCA=None, init_ORCA=None):

    batch_size = batch
    seq_len = seq

    vocab_size = config['vocab_size']
    n_embd = config['n_embd']
    n_head = config['n_head']

    # initiation phase
    if init_or_gen == 'init':

        layer = f'{model_name}-embd'
        input_ids_1 = torch.randint(0, 256, (batch_size, seq_len)).cpu()
        model = Embd(vocab_size, n_embd, batch_size, seq_len).cpu().eval()
        with torch.no_grad():
            output_1 = model(input_ids_1)

        # print(f"Saving {layer}")
        save_model(layer, model, input_ids_1, output_1, half=half,
                                        opset_version=11,      
                                        input_names=['input1'])
        if 'llama' in model_name:
            # RMSNorm
            layer = f'{model_name}-rms'
            input_ids_1 = torch.rand(batch_size, seq_len, n_embd).cpu()
            model = RMSNorm(batch_size, seq_len, n_embd).cpu().eval()
            with torch.no_grad():
                output_1 = model(input_ids_1)

            # print(f"Saving {layer}")
            save_model(layer, model, input_ids_1, output_1, half=half,
                                            opset_version=11,      
                                            input_names=['input1'])
        else:
            layer = f'{model_name}-ln'
            input_ids_1 = torch.rand(batch_size, seq_len, n_embd).cpu()
            model = LayerNorm(n_embd).cpu().eval()
            with torch.no_grad():
                output_1 = model(input_ids_1)

            # print(f"Saving {layer}")
            save_model(layer, model, input_ids_1, output_1, half=half,
                                            opset_version=11,
                                            input_names=['input1'])

        if gen_ORCA == None and init_ORCA == None:
            if 'llama' in model_name:
                layer = f'{model_name}-rattn'
                input_ids_1 = torch.rand(batch_size * seq_len, n_embd).cpu()
                model = RoPE_Attn(n_head, n_embd, batch_size, seq_len).cpu().eval()
                with torch.no_grad():
                    output_1 = model(input_ids_1)

                # print(f"Saving {layer}")
                save_model(layer, model, input_ids_1, output_1, half=half,
                                                opset_version=11,      
                                                input_names=['input1'])
            else:
                layer = f'{model_name}-attn'
                input_ids_1 = torch.rand(batch_size * seq_len, n_embd).cpu()
                model = Attn(n_head, n_embd, batch_size, seq_len).cpu().eval()
                with torch.no_grad():
                    output_1 = model(input_ids_1)

                # print(f"Saving {layer}")
                save_model(layer, model, input_ids_1, output_1, half=half,
                                                opset_version=11,      
                                                input_names=['input1'])
        else:

            layer = f'{model_name}-qkv'
            input_ids_1 = torch.rand(batch_size * seq_len, n_embd).cpu()
            model = QKV_Orca(n_embd).cpu().eval()
            with torch.no_grad():
                output_1 = model(input_ids_1)

            # print(f"Saving {layer}")
            save_model(layer, model, input_ids_1, output_1, half=half,
                                            opset_version=11,      
                                            input_names=['input1'])

        layer = f'{model_name}-proj'
        input_ids_1 = torch.rand(batch_size * seq_len, n_embd).cpu()
        input_ids_2 = torch.rand(batch_size, seq_len, n_embd).cpu() # residual
        model = Proj(n_embd, batch_size, seq_len).cpu().eval()
        with torch.no_grad():
            output_1 = model(input_ids_1, input_ids_2)

        # print(f"Saving {layer}")
        save_model(layer, model, (input_ids_1, input_ids_2), output_1, half=half,
                                        opset_version=11,      
                                        input_names=['input1', 'input2'])
        if 'llama' in model_name:
            layer = f'{model_name}-linswi'
            input_ids_1 = torch.rand(batch_size * seq_len, n_embd).cpu()
            input_ids_2 = torch.rand(batch_size, seq_len, n_embd).cpu() # residual
            model = FFNs_with_SwiGLU(batch_size, seq_len, n_embd, n_embd).cpu().eval()
            with torch.no_grad():
                output_1 = model(input_ids_1, input_ids_2)

            # print(f"Saving {layer}")
            save_model(layer, model, (input_ids_1, input_ids_2), output_1, half=half,
                                            opset_version=11,      
                                            input_names=['input1', 'input2'])
        else:
            layer = f'{model_name}-linear1'
            input_ids_1 = torch.rand(batch_size * seq_len, n_embd).cpu()
            model = FFN1_with_GELU(n_embd, batch_size, seq_len).cpu().eval()
            with torch.no_grad():
                output_1 = model(input_ids_1)

            # print(f"Saving {layer}")
            save_model(layer, model, input_ids_1, output_1, half=half,
                                            opset_version=11,      
                                            input_names=['input1'])

            layer = f'{model_name}-linear2'
            input_ids_1 = torch.rand(batch_size * seq_len, 4 * n_embd).cpu()
            input_ids_2 = torch.rand(batch_size, seq_len, n_embd).cpu() # residual
            model = FFN2(n_embd, batch_size, seq_len).cpu().eval()
            with torch.no_grad():
                output_1 = model(input_ids_1, input_ids_2)

            # print(f"Saving {layer}")
            save_model(layer, model, (input_ids_1, input_ids_2), output_1, half=half,
                                            opset_version=11,      
                                            input_names=['input1', 'input2'])

    # generation phase
    else:
        if gen_ORCA == None and init_ORCA == None:
            if 'llama' in model_name:
                layer = f'{model_name}-rgen'
                input_ids_1 = torch.rand(batch_size, 1, n_head, n_embd // n_head)
                input_ids_2 = torch.rand(batch_size, 1, n_head, n_embd // n_head)
                input_ids_3 = torch.rand(batch_size, 1, n_head, n_embd // n_head)
                input_ids_4 = torch.rand(batch_size, n_head, n_embd // n_head, seq_len)
                input_ids_5 = torch.rand(batch_size, n_head, seq_len, n_embd // n_head)
                model = RoPE_Gen(n_head, n_embd, seq_len).cpu().eval()
                with torch.no_grad():
                    output_1 = model(input_ids_1, input_ids_2, input_ids_3, input_ids_4, input_ids_5)

                # print(f"Saving {layer}")
                save_model(layer, model, (input_ids_1, input_ids_2, input_ids_3, input_ids_4, input_ids_5), output_1, half=half,
                                                opset_version=11,      
                                                input_names=['input1', 'input2', 'input3', 'input4', 'input5'])
            else:
                layer = f'{model_name}-gen'
                input_ids_1 = torch.rand(batch_size, 1, n_head, n_embd // n_head)
                input_ids_2 = torch.rand(batch_size, 1, n_head, n_embd // n_head)
                input_ids_3 = torch.rand(batch_size, 1, n_head, n_embd // n_head)
                input_ids_4 = torch.rand(batch_size, n_head, n_embd // n_head, seq_len)
                input_ids_5 = torch.rand(batch_size, n_head, seq_len, n_embd // n_head)
                model = Gen(n_head, n_embd, seq_len).cpu().eval()
                with torch.no_grad():
                    output_1 = model(input_ids_1, input_ids_2, input_ids_3, input_ids_4, input_ids_5)

                # print(f"Saving {layer}")
                save_model(layer, model, (input_ids_1, input_ids_2, input_ids_3, input_ids_4, input_ids_5), output_1, half=half,
                                                opset_version=11,      
                                                input_names=['input1', 'input2', 'input3', 'input4', 'input5'])
        else:
            
            if init_ORCA != None:
                if 'llama' in model_name:
                    for i in init_ORCA:
                        layer = f'{model_name}-rinit-{i}'
                        input_ids_1 = torch.rand(batch_size, i, n_head, n_embd // n_head).cpu()
                        input_ids_2 = torch.rand(batch_size, i, n_head, n_embd // n_head).cpu()
                        input_ids_3 = torch.rand(batch_size, i, n_head, n_embd // n_head).cpu()
                        model = RoPE_Attn_Orca(n_head, n_embd, batch_size, i).cpu().eval()
                        with torch.no_grad():
                            output_1 = model(input_ids_1, input_ids_2, input_ids_3)

                        # print(f"Saving {layer}")
                        save_model(layer, model, (input_ids_1, input_ids_2, input_ids_3), output_1, half=half,
                                                        opset_version=11,      
                                                        input_names=['input1', 'input2', 'input3'])
                else:
                    for i in init_ORCA:
                        layer = f'{model_name}-init-{i}'
                        input_ids_1 = torch.rand(batch_size, i, n_head, n_embd // n_head).cpu()
                        input_ids_2 = torch.rand(batch_size, i, n_head, n_embd // n_head).cpu()
                        input_ids_3 = torch.rand(batch_size, i, n_head, n_embd // n_head).cpu()
                        model = Attn_Orca(n_head, n_embd, batch_size, i).cpu().eval()
                        with torch.no_grad():
                            output_1 = model(input_ids_1, input_ids_2, input_ids_3)

                        # print(f"Saving {layer}")
                        save_model(layer, model, (input_ids_1, input_ids_2, input_ids_3), output_1, half=half,
                                                        opset_version=11,      
                                                        input_names=['input1', 'input2', 'input3'])
            if gen_ORCA != None:
                if 'llama' in model_name:
                    for i in gen_ORCA:
                        layer = f'{model_name}-rgen-{i}'
                        input_ids_1 = torch.rand(batch_size, 1, n_head, n_embd // n_head)
                        input_ids_2 = torch.rand(batch_size, 1, n_head, n_embd // n_head)
                        input_ids_3 = torch.rand(batch_size, 1, n_head, n_embd // n_head)
                        input_ids_4 = torch.rand(batch_size, n_head, n_embd // n_head, i)
                        input_ids_5 = torch.rand(batch_size, n_head, i, n_embd // n_head)
                        model = RoPE_Gen(n_head, n_embd, i).cpu().eval()
                        with torch.no_grad():
                            output_1 = model(input_ids_1, input_ids_2, input_ids_3, input_ids_4, input_ids_5)

                        # print(f"Saving {layer}")
                        save_model(layer, model, (input_ids_1, input_ids_2, input_ids_3, input_ids_4, input_ids_5), output_1, half=half,
                                                        opset_version=11,      
                                                        input_names=['input1', 'input2', 'input3', 'input4', 'input5'])
                else:
                    for i in gen_ORCA:
                        layer = f'{model_name}-gen-{i}'
                        input_ids_1 = torch.rand(batch_size, 1, n_head, n_embd // n_head)
                        input_ids_2 = torch.rand(batch_size, 1, n_head, n_embd // n_head)
                        input_ids_3 = torch.rand(batch_size, 1, n_head, n_embd // n_head)
                        input_ids_4 = torch.rand(batch_size, n_head, n_embd // n_head, i)
                        input_ids_5 = torch.rand(batch_size, n_head, i, n_embd // n_head)
                        model = Gen(n_head, n_embd, i).cpu().eval()
                        with torch.no_grad():
                            output_1 = model(input_ids_1, input_ids_2, input_ids_3, input_ids_4, input_ids_5)

                        # print(f"Saving {layer}")
                        save_model(layer, model, (input_ids_1, input_ids_2, input_ids_3, input_ids_4, input_ids_5), output_1, half=half,
                                                        opset_version=11,      
                                                        input_names=['input1', 'input2', 'input3', 'input4', 'input5'])
class Block(nn.Module):
    def __init__(self, config, batch, seq):
        super().__init__()
        self.vocab_size = config['vocab_size']
        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.n_layer = config['n_layer']
        self.batch = batch
        self.seq = seq
        self.ln1 = LayerNorm(self.n_embd)
        self.ln2 = LayerNorm(self.n_embd)
        self.attn = Attn(self.n_head, self.n_embd, self.batch, self.seq)
        self.proj = Proj(self.n_embd, self.batch, self.seq)
        self.ffn1 = FFN1_with_GELU(self.n_embd, self.batch, self.seq)
        self.ffn2 = FFN2(self.n_embd, self.batch, self.seq)

    def forward(self, x):
        y = self.ln1(x)
        y, k, v = self.attn(y)
        x = self.proj(y, x)
        y = self.ln2(x)
        y = self.ffn1(y)
        x = self.ffn2(y, x)
        return x


class GPT(nn.Module):
    def __init__(self, config, batch, seq):
        super().__init__()
        self.vocab_size = config['vocab_size']
        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.n_layer = config['n_layer']
        self.batch = batch
        self.seq = seq
        self.embd = Embd(self.vocab_size, self.n_embd, self.batch, self.seq)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layer):
            self.layers.append(Block(config, batch, seq))
        self.ln = LayerNorm(self.n_embd)

    def forward(self, x):
        x = self.embd(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return x
        



def export_full_model(model_name, config, batch, seq):
    
    model = GPT(config, batch, seq)
    input_ids_1 = torch.randint(0, config['vocab_size'], (batch, seq))
    output_1 = model(input_ids_1)

    save_model(model_name, model, input_ids_1, output_1,
                                                    opset_version=11,      
                                                    input_names=['input1'])

if __name__ == "__main__":

    # make full model
    config = {}
    config['vocab_size'] = 50272
    config['n_embd'] = 4096
    config['n_head'] = 32
    config['n_layer'] = 32
    export_full_model('opt-6.7b', config, 16, 256)