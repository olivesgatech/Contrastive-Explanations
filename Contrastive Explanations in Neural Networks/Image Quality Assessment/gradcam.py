import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import os
from torchvision.utils import make_grid, save_image
import PIL


from utils_GradCam import find_qualitynet_layer

class GradCAM(object):

    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None


        target_layer = find_qualitynet_layer(self.model_arch, layer_name)
            
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, input1, input2, class_idx, retain_graph=False):

        input1 = input1.unsqueeze(0).unsqueeze(0)
        input2 = input2.unsqueeze(0).unsqueeze(0)

        num, b, c, h, w = input1.size()
        self.model_arch.eval()
        self.model_arch.cuda()
        input1.cuda()
        input2.cuda()
        logit = self.model_arch((input1, input2))

        score = logit[:, logit.max(1)[-1]].squeeze()


        self.model_arch.zero_grad()
        score.backward()

        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        return saliency_map, logit

    def __call__(self, input1, input2, class_idx=None, retain_graph=False):

        return self.forward(input1, input2, class_idx, retain_graph)

class Contrast(object):

    ###Calculate Contrast map.
    def __init__(self, model_dict, verbose=False):

        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer = find_qualitynet_layer(self.model_arch, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                # self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                # print('saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, input1, input2, Q, class_idx=None, retain_graph=False):

        input1 = input1.unsqueeze(0).unsqueeze(0)
        input2 = input2.unsqueeze(0).unsqueeze(0)

        num, b, c, h, w = input1.size()
        self.model_arch.eval()
        self.model_arch.cuda()
        input1.cuda()
        input2.cuda()
        logit = self.model_arch((input1, input2))

        ce_loss = nn.MSELoss()
        im_label_as_var2 = (torch.from_numpy(Q * np.ones((1))).float())
        im_label_as_var2 = im_label_as_var2.unsqueeze(0)
        score = ce_loss(logit.cuda(), im_label_as_var2.cuda())

        self.model_arch.zero_grad()
        score.backward()

        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(32, 32), mode='bilinear', align_corners=False)

        del ce_loss, score

        return saliency_map, logit

    def __call__(self, input1, input2, contrast, class_idx=None, retain_graph=False):
        return self.forward(input1, input2, contrast, class_idx, retain_graph)