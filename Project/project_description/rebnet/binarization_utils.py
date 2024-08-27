import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn.init as init
import copy
import math
from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention, BertForSequenceClassification
class MultiLevelBinaryQuantize(nn.Module):
    def __init__(self, levels):
        super(MultiLevelBinaryQuantize, self).__init__()
        self.levels = levels
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = x / self.scale
        x = torch.clamp(x, -self.levels, self.levels)
        x = torch.round(x)
        x = x * self.scale
        return x
class QuantizeAndPruneLayer(nn.Module):
    def __init__(self, k):
        super(QuantizeAndPruneLayer, self).__init__()
        self.k = k

    def forward(self, x):
        x = self.quantize_4bit(x)
        return x

    @staticmethod
    def quantize_4bit(x):
        x = x.int() >> 4  # Truncate the lower 4 bits
        return x

    @staticmethod
    def k_top_pruning(x, k):
        values, indices = torch.topk(x, k, dim=-1)
        mask = torch.zeros_like(x).scatter_(-1, indices, 1)
        return x * mask

class QuantizedBertSelfAttention(BertSelfAttention):
    def __init__(self, config, quantize_and_prune_layer):
        super().__init__(config)
        self.quantize_and_prune_layer = quantize_and_prune_layer

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:  # Self-attention
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)

        query_layer = self.quantize_and_prune_layer(query_layer).float()
        key_layer = self.quantize_and_prune_layer(key_layer).float()

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.quantize_and_prune_layer.k_top_pruning(attention_probs, self.quantize_and_prune_layer.k)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        value_layer = self.transpose_for_scores(mixed_value_layer)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
class BertForSequenceClassificationWithPruning(BertForSequenceClassification):
    def __init__(self, config, k=10,levels=15):
        super(BertForSequenceClassificationWithPruning, self).__init__(config)
        self.bert.encoder.layer[0].attention.self = QuantizedBertSelfAttention(config, QuantizeAndPruneLayer(k))
        # self.quantize = MultiLevelBinaryQuantize(levels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[1]

        logits = self.classifier(sequence_output)
        # logits = self.quantize(logits)

        probabilities = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, probabilities
        else:
            return probabilities
def gamma_initializer(shape, gain= .5, dtype=None):
    return gain * torch.arange(shape,0,-1).float()/shape

class Round(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient is passed through unchanged
        return grad_output

class HighestPowerOf2(nn.Module):
    def __init__(self):
        super(HighestPowerOf2, self).__init__()

    def forward(self, n):
        n = n.clone().detach().requires_grad_(True)

        # Calculate the power
        p = torch.log2(n)

        # Use custom round operation
        p = Round.apply(p)
        return 2 ** p
    
class FPQuantize(nn.Module):
    def __init__(self, w, f):
        super(FPQuantize, self).__init__()
        self.w = w
        self.f = f
        self.i = w - f
        self.max_val = float(2 ** (self.i - 1) - 2 ** (-f))
        self.min_val = float(-2 ** (self.i - 1))
        self.n = float(2 ** f)

    def forward(self, x):
        x = Round.apply(x * self.n + 0.5) / self.n
        x = torch.clamp(x, min=self.min_val, max=self.max_val)
        return x


class Abs(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.abs()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_variables
        return grad_output * x.sign()

class Binarize(Function):
    @staticmethod
    def forward(ctx, x):
        ctx._mask = (x.ge(-1) * x.le(1))
        clipped = torch.clamp(x, -1, 1)
        rounded = torch.sign(clipped).clone().detach()
        sp = (rounded - clipped.clone().detach()).detach()
        sp.requires_grad = False
        return clipped + sp
    
    @staticmethod
    def backward(ctx, grad_output):
        mask = torch.autograd.Variable(ctx._mask.type_as(grad_output.data))
        return grad_output * mask

class FunRes(Function):
    @staticmethod
    def forward(ctx, x, out_bin, out):
        o_bin = out_bin + out
        resid = x - out
        return o_bin, resid
    
    @staticmethod
    def backward(ctx, grad_o_bin, grad_resid):
        # Initialize gradients for x, out_bin, and out
        grad_x = grad_out_bin = grad_out = None

        if ctx.needs_input_grad[0]:
            # Compute gradient with respect to x
            grad_x = grad_resid
        if ctx.needs_input_grad[1]:
            # Compute gradient with respect to out_bin
            grad_out_bin = grad_o_bin
        if ctx.needs_input_grad[2]:
            # Compute gradient with respect to out
            grad_out = grad_o_bin

        # Return the gradients, each with respect to its corresponding input
        return grad_x, grad_out_bin, grad_out

class MLBinarize(nn.Module):
    def __init__(self, levels, gamma):
        super(MLBinarize, self).__init__()
        self.levels = levels
        self.gamma = gamma

        self.highest_power_of_2 = HighestPowerOf2()

    def forward(self, x):
        resid = x
        out_bin = 0
        for l in range(self.levels):
            out = Binarize.apply(resid) * self.highest_power_of_2(Abs.apply(self.gamma[l]))
            out_bin, resid = FunRes.apply(resid, out_bin, out)
        return out_bin
        
class ResidualSign(nn.Module):
    def __init__(self, levels=1):
        super(ResidualSign, self).__init__()
        self.levels = levels
        ars = np.arange(self.levels) + 1.0
        ars = ars[::-1]
        means = ars / np.sum(ars)
        self.means = nn.Parameter(torch.tensor(means, dtype=torch.float32))

        self.param = {
            'levels': self.levels
        }
        self.type = 'res'

    def forward(self, x):
        if self.means is None:
            raise ValueError("Call set_means(X) before using the layer.")
        
        resid = x
        out_bin = 0
        for l in range(self.levels):
            out = Binarize.apply(resid) * Abs.apply(self.means[l])
            out_bin, resid = FunRes.apply(resid, out_bin, out)
        
        return out_bin
    
    def _save_to_state_dict(self, destination=None, prefix='', keep_vars=False):
        # Save additional attributes along with the state
        destination[prefix + 'param'] = self.param
        destination[prefix + 'type'] = self.type
        super()._save_to_state_dict(destination, prefix, keep_vars)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.param = state_dict.pop(prefix + 'param', None)
        self.type = state_dict.pop(prefix + 'type', None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    def set_means(self, X):
        means = np.zeros((self.levels))
        means[0] = 1
        resid = np.clip(X, -1, 1)
        approx = 0
        for l in range(self.levels):
            m = np.mean(np.absolute(resid))
            out = np.sign(resid) * m
            approx = approx + out
            resid = resid - out
            means[l] = m
            err = np.mean((approx - np.clip(X, -1, 1))**2)

        means = means / np.sum(means)

class BinaryInput(nn.Module):
    def __init__(self, levels=2):
        super(BinaryInput, self).__init__()
        self.levels = levels

        ars = np.arange(self.levels) + 1.0
        ars = ars[::-1]
        means = ars / np.sum(ars)
        self.means = nn.Parameter(torch.tensor(means, dtype=torch.float32), requires_grad=True)

        self.param = {
            'levels': self.levels
        }
        self.type = 'input'

    def forward(self, x):
        bn_module = MLBinarize(self.levels, self.means)
        return bn_module(x)
        
    def _save_to_state_dict(self, destination=None, prefix='', keep_vars=False):
        # Save additional attributes along with the state
        destination[prefix + 'param'] = self.param
        destination[prefix + 'type'] = self.type
        super()._save_to_state_dict(destination, prefix, keep_vars)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.param = state_dict.pop(prefix + 'param', None)
        self.type = state_dict.pop(prefix + 'type', None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

class FunBinaryWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, gamma):
        ctx.save_for_backward(weight, gamma)
        clamped_w = gamma * weight
        return clamped_w
    
    @staticmethod
    def backward(ctx, grad_output):
        weight, gamma = ctx.saved_tensors

        grad_weight = grad_output * gamma  # Gradient with respect to weight
        grad_gamma = torch.sum(grad_output * weight)  # Gradient with respect to gamma

        return grad_weight, grad_gamma

class BinaryLinear(nn.Linear):
    def __init__(self, n_in, n_out, w_bits=2, a_bits=2):
        super(BinaryLinear, self).__init__(n_in, n_out, bias=False)
        self.n_in = n_in
        self.n_out = n_out
        self.w_levels = w_bits
        self.a_levels = a_bits

        stdv = 1 / np.sqrt(self.n_in)
        w = np.random.normal(loc=0.0, scale=stdv, size=(self.n_in, self.n_out)).astype(np.float32)
        self.weight = nn.Parameter(torch.tensor(w))
        self.gamma = nn.Parameter(gamma_initializer(self.w_levels, 0.5), requires_grad=True)

        self.mlb = MLBinarize(self.w_levels, self.gamma)
        self.amlb = ResidualSign(a_bits)

    def set_weight(self, weight):
        self.weight = nn.Parameter(torch.tensor(weight))
        
    def forward(self, x):
        if self.a_levels > 0:
            x = self.amlb(x)
        clamped_w = self.mlb(self.weight)
        out = F.linear(x, clamped_w, None)
        return out


class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1,
                groups=1, dilation=1, padding_mode='zeros', levels=1):

        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                    dilation=dilation, groups=groups, padding_mode=padding_mode, bias=False)

        self.w_levels = levels

        if type(kernel_size) is int:
            self.k = (kernel_size, kernel_size)
        else:
            self.k = kernel_size
        
        stdv = 1 / np.sqrt(self.k[0] * self.k[1] * self.in_channels)
        w = np.random.normal(loc=0.0, scale=stdv, size=(self.out_channels, self.in_channels, self.k[0], self.k[1])).astype(np.float32)
        self.weight = nn.Parameter(torch.tensor(w))
        self.gamma = nn.Parameter(gamma_initializer(self.w_levels, 0.5), requires_grad=True)

        self.mlb = MLBinarize(self.w_levels, self.gamma)

        self.param = {
            self.w_levels
        }
        self.type = 'conv2d'

    def set_weight(self, weight):
        self.weight = nn.Parameter(torch.tensor(weight))

    def forward(self, x):
        clamped_w = self.mlb(self.weight)
        out = F.conv2d(x, clamped_w, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
        return out
    
    def _save_to_state_dict(self, destination=None, prefix='', keep_vars=False):
        # Save additional attributes along with the state
        destination[prefix + 'param'] = self.param
        destination[prefix + 'type'] = self.type
        super()._save_to_state_dict(destination, prefix, keep_vars)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.param = state_dict.pop(prefix + 'param', None)
        self.type = state_dict.pop(prefix + 'type', None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

def add_quant_op(module, layer_counter, a_bits=8, w_bits=8):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            layer_counter[0] += 1
            quant_linear = BinaryLinear(child.in_features, child.out_features,
                                        a_bits=a_bits, w_bits=w_bits)

            quant_linear.weight.data = child.weight
            module._modules[name] = quant_linear
        else:
            add_quant_op(child, layer_counter, a_bits=a_bits, w_bits=w_bits)


def prepare(model, inplace=False, a_bits=8, w_bits=8):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    add_quant_op(model, layer_counter, a_bits=a_bits, w_bits=w_bits)
    return model
model = BertForSequenceClassificationWithPruning.from_pretrained('bert-base-uncased', k=10)
model_2 = prepare(model)
print(model_2)
print(model)