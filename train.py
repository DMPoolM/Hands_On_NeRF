import os
import cv2
import json
import torch
import imageio
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import DatasetProvider, NeRFDataset
from utils import NeRF
from utils import sample_rays, sample_viewdirs, predict_to_rgb, sample_pdf

def render_rays(coarse, fine, raydirs, rayoris, sample_z_values, num_samples2, white_background):
    rays, z_values = sample_rays(raydirs, rayoris, sample_z_values)
    view_dirs = sample_viewdirs(raydirs)
    sigma, rgb = coarse(rays, view_dirs)
    sigma = sigma.squeeze(dim=-1)
    rgb1, _, _, weights = predict_to_rgb(sigma, rgb, z_values, raydirs, white_background)

    z_values_mid = .5 * (z_values[..., 1:] + z_values[..., :-1])
    z_samples = sample_pdf(z_values_mid, weights[..., 1:-1], num_samples2, det=True)
    z_samples = z_samples.detach()
    z_values, _ = torch.sort(torch.cat([z_values, z_samples], -1), -1)
    rays = rayoris[..., None, :] + raydirs[..., None, :] * z_values[..., :, None]
    sigma, rgb = fine(rays, view_dirs)
    sigma = sigma.squeeze(dim=-1)
    # Predict RGB values for the fine model
    rgb2, _, _, _ = predict_to_rgb(sigma, rgb, z_values, raydirs, white_background)

    return rgb1, rgb2


if __name__ == "__main__":
    root = 'nerf_synthetic/lego/'
    transforms_file = 'transforms_train.json'
    half_resolution = False
    provider = DatasetProvider(root, transforms_file, half_resolution)

    batch_size = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainset = NeRFDataset(provider, batch_size, device=device)

    x_posendim = 10
    view_posendim = 4
    coarse = NeRF(x_posendim = x_posendim, view_posendim = view_posendim).to(device)
    fine = NeRF(x_posendim = x_posendim, view_posendim = view_posendim).to(device)
    params = list(coarse.parameters())    
    params.extend(list(fine.parameters()))

    lrate = 5e-4
    optimizer = optim.Adam(params, lr=lrate)
    loss_fn = nn.MSELoss()

    num_epochs = 10000
    for epoch in range(num_epochs):
        for i in range(len(trainset)):
            ray_dirs, ray_oris, pixels = trainset[i]

            num_samples1 = 64
            num_samples2 = 128
            white_background = True
            sample_z_values = torch.linspace(
                2.0, 6.0, num_samples1, device=device
            ).expand(ray_dirs.shape[0], num_samples1)

            rgb1, rgb2 = render_rays(
                coarse, fine, ray_dirs, ray_oris,
                sample_z_values, num_samples2, white_background
            )
            
            loss = loss_fn(rgb1, pixels) + loss_fn(rgb2, pixels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
