# This code provides contrastive explanations for the application of Image Quality Assessment
# The base IQA.py and checkpoints are downloaded from https://github.com/lidq92/WaDIQaM
# Demo images are taken from TID 2013 database

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from IQA import RandomCropPatches, NonOverlappingCropPatches, FRnet
import numpy as np
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid, save_image

from utils_GradCam import visualize_cam, Normalize
from gradcam import GradCAM, Contrast

def CreateImage(im, ref=None, patch_size=32):

    w, h = ref.size
    width_patches = int(w/32)
    height_patches = int(h/32)

    img = torch.zeros(0)
    img_col = torch.zeros(0)

    im = im.data.cpu()
    w_patch = 0

    for ii in range(0, len(im)):

        temp = im[ii]
        w_patch+=1
        img_col = torch.cat((img_col, temp), dim = 1)

        if w_patch == width_patches:

            w_patch = 0
            img = torch.cat((img, img_col), dim = 0)
            img_col = torch.zeros(0)

    return img

if __name__ == "__main__":

    parser = ArgumentParser(description='PyTorch WaDIQaM-FR test')
    parser.add_argument("--dist_path", type=str, default='Images/Distorted_Images/i21_21_5.bmp',
                        help="distorted image path.")
    parser.add_argument("--ref_path", type=str, default='Images/Reference_Images/I21.BMP',
                        help="reference image path.")
    parser.add_argument("--model_file", type=str, default='checkpoints/WaDIQaM-FR-KADID-10K-EXP1000-5-lr=0.0001-bs=4',
                        help="model file (default: checkpoints/WaDIQaM-FR-KADID-10K-EXP1000-5-lr=0.0001-bs=4)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FRnet(weighted_average=True).to(device)
    model.load_state_dict(torch.load(args.model_file))
    model.eval()
    model_dict = dict(type='qualitynet', arch=model, layer_name='10', input_size=(224, 224))

    im = Image.open(args.dist_path).convert('RGB')
    ref = Image.open(args.ref_path).convert('RGB')

    model_gradcam = GradCAM(model_dict, False)
    model_contrast = Contrast(model_dict, False)

    w, h = im.size

    stride = 4
    w, h = im.size
    contrast_im = Image.new('L', (w,h))#(stride, w))
    gradcam_im = Image.new('L', (w,h))

    patch_size = 32
    width_patches = int(w / stride)
    img = torch.zeros(0)
    img_all = Image.new('RGB', (w, h))
    w_patch = 0
    count = 0

    contrast = 0.75 # Definition of Q - `Why P, rather than Q?' or in other words, `Why score, rather than 1?'

    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):

            count+=1
            print(count)
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patch_ref = to_tensor(ref.crop((j, i, j + patch_size, i + patch_size)))

            contrast_patches, _ = model_contrast(patch.to(device), patch_ref.to(device), contrast)
            gradcam_patches, _ = model_gradcam(patch.to(device), patch_ref.to(device))

            temp_contrast = contrast_patches.data.cpu().squeeze(0).squeeze(0)
            np_patch = np.asarray((temp_contrast*255))
            img_patch = Image.fromarray(np.uint8(np_patch))
            mask_im_blur = img_patch.filter(ImageFilter.GaussianBlur(5))
            contrast_im.paste(img_patch, (j, i),  mask_im_blur)

            temp_gradcam = gradcam_patches.data.cpu().squeeze(0).squeeze(0)
            np_patch = np.asarray((temp_gradcam * 255))
            img_patch = Image.fromarray(np.uint8(np_patch))
            mask_im_blur = img_patch.filter(ImageFilter.GaussianBlur(5))
            gradcam_im.paste(img_patch, (j, i), mask_im_blur)

    data = NonOverlappingCropPatches(im, ref)
    dist_patches = data[0].unsqueeze(0).to(device)
    ref_patches = data[1].unsqueeze(0).to(device)
    score = model((dist_patches, ref_patches))

    print('The quality of distorted image is ' + str(score.item()))

    contrast_im = np.array(contrast_im)
    contrast_im = torch.from_numpy(contrast_im).unsqueeze(0).unsqueeze(0)
    ref_torch = torch.from_numpy(np.asarray(ref)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    _, result_contrast = visualize_cam(contrast_im, ref_torch)
    save_image(result_contrast, 'Why ' + str(score.item()) + ', rather than ' + str(contrast) + '?.png')

    gradcam_im = np.array(gradcam_im)
    gradcam_im = torch.from_numpy(gradcam_im).unsqueeze(0).unsqueeze(0)
    ref_torch = torch.from_numpy(np.asarray(ref)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    _, result_gradcam = visualize_cam(gradcam_im, ref_torch)
    save_image(result_gradcam, 'Why ' + str(score.item()) + '?.png')
