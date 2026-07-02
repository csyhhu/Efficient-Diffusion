import torch
import torch.nn as nn
import torch.nn.functional as F

# supports both package import and direct script execution
try:
    from ..quant_utils.quantization import AsymmetricQuantization
except ImportError:
    import sys, os
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _src)
    from quant_utils.quantization import AsymmetricQuantization


class QuantizedMultiHeadAttention(nn.MultiheadAttention):

    def __init__(self, *args, **kwargs):
        self.bitW = kwargs.pop('bitW', 8)
        self.bitA = kwargs.pop('bitA', 8)
        self.bitG = kwargs.pop('bitG', 8)
        self.parameter_quantization_axis = kwargs.pop('parameter_quantization_axis', -1)
        self.activation_quantization_axis = kwargs.pop('activation_quantization_axis', None)
        self.layer_prefix = kwargs.pop('layer_prefix', None)
        super().__init__(*args, **kwargs)

        # in_proj_weight shape: [3*embed_dim, embed_dim]  (Q, K, V concatenated)
        # out_proj.weight shape:   [embed_dim, embed_dim]
        embed_dim = self.embed_dim
        if self.parameter_quantization_axis is None:
            # Q projection
            self.q_proj_left_threshold = nn.Parameter(torch.zeros(()))
            self.q_proj_right_threshold = nn.Parameter(torch.ones(()))
            # K projection
            self.k_proj_left_threshold = nn.Parameter(torch.zeros(()))
            self.k_proj_right_threshold = nn.Parameter(torch.ones(()))
            # V projection
            self.v_proj_left_threshold = nn.Parameter(torch.zeros(()))
            self.v_proj_right_threshold = nn.Parameter(torch.ones(()))
            # out projection
            self.out_proj_left_threshold = nn.Parameter(torch.zeros(()))
            self.out_proj_right_threshold = nn.Parameter(torch.ones(()))
        else:
            # Q/K/V each is [embed_dim, embed_dim] from in_proj_weight; both axes are embed_dim
            proj_size = embed_dim
            # Q projection
            self.q_proj_left_threshold = nn.Parameter(torch.zeros(proj_size))
            self.q_proj_right_threshold = nn.Parameter(torch.ones(proj_size))
            # K projection
            self.k_proj_left_threshold = nn.Parameter(torch.zeros(proj_size))
            self.k_proj_right_threshold = nn.Parameter(torch.ones(proj_size))
            # V projection
            self.v_proj_left_threshold = nn.Parameter(torch.zeros(proj_size))
            self.v_proj_right_threshold = nn.Parameter(torch.ones(proj_size))
            # Output projection
            out_proj_size = self.out_proj.weight.shape[-3 - self.parameter_quantization_axis]
            self.out_proj_left_threshold = nn.Parameter(torch.zeros(out_proj_size))
            self.out_proj_right_threshold = nn.Parameter(torch.ones(out_proj_size))

        # print(f"Initialize quantized q_proj threshold with shape: {self.q_proj_left_threshold.shape}")
        # print(f"Initialize quantized k_proj threshold with shape: {self.k_proj_left_threshold.shape}")
        # print(f"Initialize quantized v_proj threshold with shape: {self.v_proj_left_threshold.shape}")
        # print(f"Initialize quantized out_proj threshold with shape: {self.out_proj_left_threshold.shape}")

        self.activation_left_threshold = nn.Parameter(torch.tensor(-1.0))
        self.activation_right_threshold = nn.Parameter(torch.tensor(1.0))
        # print(f"Initialize quantized activation threshold with shape: {self.activation_left_threshold.shape}")

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False, quantization_error_info=None):

        # replicate parent's batch_first handling  →  canonical shape [seq, batch, embed]
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # -------------------------------------------
        # 1. Quantize Q / K / V inputs
        # -------------------------------------------
        query = AsymmetricQuantization.apply(
            query, self.bitA,
            self.activation_left_threshold, self.activation_right_threshold,
            self.activation_quantization_axis,
            quantization_error_info, f"{self.layer_prefix}.query.input"
        )

        key = AsymmetricQuantization.apply(
            key, self.bitA,
            self.activation_left_threshold, self.activation_right_threshold,
            self.activation_quantization_axis,
            quantization_error_info, f"{self.layer_prefix}.key.input"
        )

        value = AsymmetricQuantization.apply(
            value, self.bitA,
            self.activation_left_threshold, self.activation_right_threshold,
            self.activation_quantization_axis,
            quantization_error_info, f"{self.layer_prefix}.value.input"
        )

        # -------------------------------------------
        # 2. Quantize Q / K / V / out projection weights
        # -------------------------------------------
        embed_dim = self.embed_dim
        num_heads = self.num_heads
        head_dim = embed_dim // num_heads

        q_weight, k_weight, v_weight = self.in_proj_weight.split(embed_dim, dim=0)
        q_weight = AsymmetricQuantization.apply(
            q_weight, self.bitW,
            self.q_proj_left_threshold, self.q_proj_right_threshold,
            self.parameter_quantization_axis,
            quantization_error_info, f"{self.layer_prefix}.query.weight"
        )
        k_weight = AsymmetricQuantization.apply(
            k_weight, self.bitW,
            self.k_proj_left_threshold, self.k_proj_right_threshold,
            self.parameter_quantization_axis,
            quantization_error_info, f"{self.layer_prefix}.key.weight"
        )
        v_weight = AsymmetricQuantization.apply(
            v_weight, self.bitW,
            self.v_proj_left_threshold, self.v_proj_right_threshold,
            self.parameter_quantization_axis,
            quantization_error_info, f"{self.layer_prefix}.value.weight"
        )

        out_proj_weight = AsymmetricQuantization.apply(
            self.out_proj.weight, self.bitW,
            self.out_proj_left_threshold, self.out_proj_right_threshold,
            self.parameter_quantization_axis,
            quantization_error_info, f"{self.layer_prefix}.output.weight"
        )

        # -------------------------------------------
        # 3. Q / K / V projection  (manual, no in_proj cat)
        # -------------------------------------------
        q_bias = k_bias = v_bias = None
        if self.in_proj_bias is not None:
            q_bias, k_bias, v_bias = self.in_proj_bias.split(embed_dim, dim=0)

        q = F.linear(query, q_weight, q_bias)
        k = F.linear(key, k_weight, k_bias)
        v = F.linear(value, v_weight, v_bias)

        tgt_len, bsz = q.shape[0], q.shape[1]
        src_len = k.shape[0]

        # reshape → [batch, num_heads, seq, head_dim]
        q = q.view(tgt_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
        v = v.view(src_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)

        # --- add_bias_kv (optional learned bias appended to K/V) ---
        bias_k = self.bias_k if hasattr(self, 'bias_k') and self.bias_k is not None else None
        bias_v = self.bias_v if hasattr(self, 'bias_v') and self.bias_v is not None else None
        if bias_k is not None and bias_v is not None:
            # bias_k/v stored by nn.MultiheadAttention: broadcast to [batch, num_heads, 1, head_dim]
            if bias_k.dim() == 5:  # [1, 1, 1, num_heads, head_dim] style
                bias_k = bias_k.reshape(1, num_heads, 1, head_dim).expand(bsz, -1, 1, -1)
                bias_v = bias_v.reshape(1, num_heads, 1, head_dim).expand(bsz, -1, 1, -1)
            elif bias_k.dim() == 3:  # [num_heads, 1, head_dim] style
                bias_k = bias_k.unsqueeze(0).expand(bsz, -1, -1, -1)
                bias_v = bias_v.unsqueeze(0).expand(bsz, -1, -1, -1)
            k = torch.cat([k, bias_k], dim=2)
            v = torch.cat([v, bias_v], dim=2)
            src_len = src_len + 1
            # pad attn_mask to match the extra key token
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = F.pad(attn_mask, (0, 1))
                elif attn_mask.dim() == 3:
                    attn_mask = F.pad(attn_mask, (0, 1))
                elif attn_mask.dim() == 4:
                    attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # --- add_zero_attn ---
        if self.add_zero_attn:
            zero_attn = torch.zeros(bsz, num_heads, 1, head_dim,
                                    dtype=k.dtype, device=k.device)
            k = torch.cat([k, zero_attn], dim=2)
            v = torch.cat([v, zero_attn], dim=2)
            src_len = src_len + 1
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = F.pad(attn_mask, (0, 1))
                elif attn_mask.dim() == 3:
                    attn_mask = F.pad(attn_mask, (0, 1))
                elif attn_mask.dim() == 4:
                    attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # --- merge key_padding_mask into attn_mask ---
        # F.scaled_dot_product_attention does NOT accept key_padding_mask
        if key_padding_mask is not None:
            # key_padding_mask: [batch, src_len], True = ignore
            kpm = torch.zeros(bsz, 1, 1, src_len, dtype=q.dtype, device=q.device)
            kpm.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_mask = kpm if attn_mask is None else attn_mask + kpm

        # -------------------------------------------
        # 4. Attention computation
        # -------------------------------------------
        dropout_p = self.dropout if self.training else 0.0
        if need_weights:
            # --- manual path: need to return attention weights ---
            scale = head_dim ** -0.5
            attn_weights_out = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                attn_weights_out = attn_weights_out + attn_mask
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(tgt_len, src_len, device=q.device, dtype=torch.bool), diagonal=1)
                attn_weights_out.masked_fill_(causal_mask, float('-inf'))
            attn_weights_out = F.softmax(attn_weights_out, dim=-1)
            if dropout_p > 0:
                attn_weights_out = F.dropout(attn_weights_out, p=dropout_p, training=self.training)
            attn_output = torch.matmul(attn_weights_out, v)
            # prepare returned weights
            if average_attn_weights:
                attn_weights_for_return = attn_weights_out.detach().mean(dim=1)
            else:
                attn_weights_for_return = attn_weights_out.detach()
        else:
            # --- fast path: F.scaled_dot_product_attention ---
            # attn_mask: broadcastable to [batch, num_heads, tgt, src] or 2D/3D/4D
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=dropout_p, is_causal=is_causal)
            attn_weights_for_return = None

        # reshape back: [batch, num_heads, tgt, head_dim] → [tgt, batch, embed_dim]
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)

        # -------------------------------------------
        # 5. Quantize attention output BEFORE out_proj
        # -------------------------------------------
        attn_output = AsymmetricQuantization.apply(
            attn_output, self.bitA,
            self.activation_left_threshold, self.activation_right_threshold,
            self.activation_quantization_axis,
            quantization_error_info, f"{self.layer_prefix}.attn.output")

        # -------------------------------------------
        # 6. Output projection
        # -------------------------------------------
        attn_output = F.linear(attn_output, out_proj_weight, self.out_proj.bias)

        # transpose back for batch_first
        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        if need_weights:
            return attn_output, attn_weights_for_return
        else:
            return (attn_output,)


if __name__ == "__main__":

    import torch

    batch_size = 4
    seq_len = 10
    embed_dim = 32
    num_heads = 4
    quantization_error_info = {}

    query = torch.randn(seq_len, batch_size, embed_dim)
    key = torch.randn(seq_len, batch_size, embed_dim)
    value = torch.randn(seq_len, batch_size, embed_dim)

    mha = QuantizedMultiHeadAttention(embed_dim, num_heads, bitW=8, bitA=8, layer_prefix="base")
    attn_output, attn_weights = mha(query, key, value, quantization_error_info=quantization_error_info)
    print("attn_output shape:", attn_output.shape)

    loss = torch.nn.MSELoss()(attn_output, torch.randn_like(attn_output))
    loss.backward()
    print("loss:", loss.item())

    # print("q_proj_left_threshold.grad  shape:", mha.q_proj_left_threshold.grad.shape)
    # print("q_proj_right_threshold.grad shape:", mha.q_proj_right_threshold.grad.shape)
    # print("k_proj_left_threshold.grad  shape:", mha.k_proj_left_threshold.grad.shape)
    # print("k_proj_right_threshold.grad shape:", mha.k_proj_right_threshold.grad.shape)
    # print("v_proj_left_threshold.grad  shape:", mha.v_proj_left_threshold.grad.shape)
    # print("v_proj_right_threshold.grad shape:", mha.v_proj_right_threshold.grad.shape)
    # print("out_proj_left_threshold.grad  shape:", mha.out_proj_left_threshold.grad.shape)
    # print("out_proj_right_threshold.grad shape:", mha.out_proj_right_threshold.grad.shape)
    # print("activation_left_threshold.grad  shape:", mha.activation_left_threshold.grad.shape)
    # print("activation_right_threshold.grad shape:", mha.activation_right_threshold.grad.shape)
    
    for key, value in quantization_error_info.items():
        print(key, value.shape)

