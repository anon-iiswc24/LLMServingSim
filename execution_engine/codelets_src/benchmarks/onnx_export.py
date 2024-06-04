# SPDX-License-Identifier: Apache-2.0

import torch
import onnxruntime
import onnx
from onnx import numpy_helper
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
from onnxconverter_common import float16

import numpy as np
import os

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut          | save_name
MODELS = [
    # (GPT2Model, GPT2Tokenizer, 'gpt2', 'gpt2'),
    (GPT2LMHeadModel, GPT2Tokenizer, 'gpt2', 'gpt2-lm-head'),
]
data_dir = 'test_data_set_0'


# KV cache modeled GPT2
class GPT2(torch.nn.Module):
    def __init__(self, batch, past_len):
        super(GPT2, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
        self.batch = batch
        self.past_len = past_len

    def forward(self, input_ids):   
        past_key_values = tuple(tuple(torch.ones([self.batch, 12, self.past_len, 64], dtype=torch.float32) for _ in range(2)) for _ in range(12))
        output = self.model(input_ids=input_ids, past_key_values=past_key_values)

            # past_key_values shape
            # 0 - 11 tuple
            # 0 - 1 tuple
            # [batch, 12, past_len, 64] tensor
        return output

# Only Attention part of GPT2 with KV cache
class Attn(torch.nn.Module):
    def __init__(self, batch, cache_len):
        super(Attn, self).__init__()
        self.cache_len = cache_len
        # add softmax layer of 16k
        if (cache_len + 1) % 16 == 0:
            self.act = torch.nn.Softmax(dim=-1)
        else:
            self.act = torch.nn.Identity()
        self.batch = batch

    def forward(self, T):
        # q,k,v = T.split(768, dim=2)

        # q = q.reshape(-1, 1, self.batch, 64).transpose(1,2)
        q = T.transpose(1,2)
        # k = torch.cat((k.reshape(-1, 1, self.batch, 64), torch.ones(12, self.cache_len, self.batch, 64)), dim=1).permute(0,2,3,1)
        k = torch.cat((T.permute(0,2,3,1), torch.ones(12, self.batch, 64, self.cache_len)), dim=3)
        # v = torch.cat((v.reshape(-1, 1, self.batch, 64), torch.ones(12, self.cache_len, self.batch, 64)), dim=1).transpose(1,2)
        v = torch.cat((T.transpose(1,2), torch.ones(12, self.batch, self.cache_len, 64)), dim=2)
        attn_score = self.act(torch.div(torch.matmul(q, k), (k.shape[-2])))
        res = torch.matmul(attn_score, v)
        return res


def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


def update_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else update_flatten_list(i, res_list)
    return res_list


def to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x


def save_tensor_proto(file_path, name, data):
    tp = numpy_helper.from_array(data)
    tp.name = name

    with open(file_path, 'wb') as f:
        f.write(tp.SerializeToString())


def save_data(test_data_dir, prefix, names, data_list):
    if isinstance(data_list, torch.autograd.Variable) or isinstance(data_list, torch.Tensor):
        data_list = [data_list]
    for i, d in enumerate(data_list):
        d = d.data.cpu().numpy()
        save_tensor_proto(os.path.join(test_data_dir, '{0}_{1}.pb'.format(prefix, i)), names[i], d)


def save_model(name, model, inputs, outputs, input_names=None, output_names=None, half=False, **kwargs):
    if hasattr(model, 'train'):
        model.train(False)
    dir = './'
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(dir, 'models')# + name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    inputs_flatten = flatten(inputs)
    inputs_flatten = update_flatten_list(inputs_flatten, [])
    outputs_flatten = flatten(outputs)
    outputs_flatten = update_flatten_list(outputs_flatten, [])
    if input_names is None:
        input_names = []
        for i, _ in enumerate(inputs_flatten):
            input_names.append('input' + str(i+1))
    else:
        np.testing.assert_equal(len(input_names), len(inputs_flatten),
                                "Number of input names provided is not equal to the number of inputs.")

    if output_names is None:
        output_names = []
        for i, _ in enumerate(outputs_flatten):
            output_names.append('output' + str(i+1))
    else:
        np.testing.assert_equal(len(output_names), len(outputs_flatten),
                                "Number of output names provided is not equal to the number of output.")

    model_dir = os.path.join(dir, name + '.onnx')
    if isinstance(model, torch.jit.ScriptModule):
        torch.onnx._export(model, inputs, model_dir, verbose=False, input_names=input_names,
                           output_names=output_names, example_outputs=outputs, **kwargs)
    else:
        torch.onnx.export(model, inputs, model_dir, verbose=False, input_names=input_names,
                          output_names=output_names, **kwargs)

    test_data_dir = os.path.join(dir, data_dir)
    # if not os.path.exists(test_data_dir):
    #     os.makedirs(test_data_dir)

    # save_data(test_data_dir, "input", input_names, inputs_flatten)
    # save_data(test_data_dir, "output", output_names, outputs_flatten)

    # if half:
    #     model = onnx.load(model_dir)
    #     model_fp16 = float16.convert_float_to_float16(model)
    #     onnx.save(model_fp16, model_dir)

    return model_dir, test_data_dir


def inference(file, inputs, outputs):
    inputs_flatten = flatten(inputs)
    inputs_flatten = update_flatten_list(inputs_flatten, [])
    outputs_flatten = flatten(outputs)
    outputs_flatten = update_flatten_list(outputs_flatten, [])

    # Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
    # other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
    # based on the build flags) when instantiating InferenceSession.
    # For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
    # onnxruntime.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
    sess = onnxruntime.InferenceSession(file)
    ort_inputs = dict((sess.get_inputs()[i].name, to_numpy(input)) for i, input in enumerate(inputs_flatten))
    res = sess.run(None, ort_inputs)

    if outputs is not None:
        print("== Checking model output ==")
        [np.testing.assert_allclose(to_numpy(output), res[i], rtol=1e-03, atol=1e-05) for i, output in enumerate(outputs_flatten)]
        print("== Done ==")

