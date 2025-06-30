import os
import cv2
import json
import torch
import imageio
import numpy as np
import torch.nn as nn

class DatasetProvider:
    def __init__(self, root, transforms_file, half_resolution=True):

        self.meta = json.load(open(os.path.join(root, transforms_file), 'r'))
        self.root = root
        self.frames = self.meta['frames']
        self.images = []
        self.poses = []
        self.camera_angle_x = self.meta['camera_angle_x']

        for frame in self.frames:
            image_file = os.path.join(self.root, frame['file_path'] + '.png')
            image = imageio.imread(image_file)
            if half_resolution:
                image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            self.images.append(image)
            self.poses.append(frame['transform_matrix'])

        self.poses = np.stack(self.poses).astype(np.float32)
        self.images = (np.stack(self.images) / 255.0).astype(np.float32)
        self.width = self.images.shape[2]
        self.height = self.images.shape[1]
        self.focal = 0.5 * self.width / np.tan(0.5 * self.camera_angle_x)

        alpha = self.images[..., [3]]
        rgb = self.images[..., :3]
        self.images = rgb * alpha + (1 - alpha)

class Embedder(nn.Module): #Fourier positional encoding
    def __init__(self, positional_encoding_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positional_encoding_dim = positional_encoding_dim
    def forward(self, x):
        res = [x]
        for i in range(self.positional_encoding_dim):
            for fn in [torch.sin, torch.cos]:
                res.append(fn((2.0 ** i) * x))
        return torch.cat(res, -1)


class ViewDependentHead(nn.Module):
    def __init__(self, n_input, n_view):
        super().__init__()
        # Define the layers for the view-dependent head
        self.feature = nn.Linear(n_input, n_input)  # A shared feature layer before view direction processing
        self.alpha = nn.Linear(n_input, 1)
        self.view_fc = nn.Linear(n_input + n_view, n_input // 2)
        self.rgb = nn.Linear(n_input // 2, 3)  # Output RGB values

    def forward(self, x, view_dirs):
        feature = self.feature(x)
        sigma = self.alpha(x).relu()
        feature = torch.cat([feature, view_dirs], -1)
        feature = self.view_fc(feature).relu()
        rgb = self.rgb(feature).sigmoid()

        return rgb, sigma


class NoViewDirHead(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.head = nn.Linear(n_input, n_output)

    def forward(self, x, view_dirs):
        x = self.head(x)
        rgb = x[..., :3].sigmoid()
        sigma = x[..., 3].relu()

        return rgb, sigma

class NeRFDataset:
    def __init__(self, provider, batch_size=1024, device='cuda'):

        self.images = torch.from_numpy(provider.images).to(device)
        self.poses = torch.from_numpy(provider.poses).to(device)
        self.focal = torch.tensor(provider.focal, dtype=torch.float32, device=device)
        self.width = provider.width
        self.height = provider.height
        self.batch_size = batch_size
        self.num_images = len(self.images)
        self.precrop_iters = 500
        self.precrop_frac = 0.5
        self.device = device

        self.xs = torch.arange(self.width, device=device)
        self.ys = torch.arange(self.height, device=device)

        self.ray_oris, self.ray_dirs = self.get_rays(self.poses)
        self.pixels = self.images.reshape(self.num_images, -1, 3)


    def get_rays(self, poses):
        x, y = torch.meshgrid(self.xs, self.ys, indexing='xy')
        x = x.flatten()
        y = y.flatten()

        # get ray directions
        ray_dirs = torch.stack(
            [
                (x - self.width / 2) / self.focal,
                -(y - self.height / 2) / self.focal,
                -torch.ones_like(x)
            ], dim=-1
        )

        # rotate ray directions from camera frame to world frame
        ray_dirs = torch.einsum('nij,mj->nmi', poses[:, :3, :3], ray_dirs)
        ray_oris = poses[:, :3, 3]

        return ray_oris, ray_dirs

    def __getitem__(self, idx):
        ray_oris = self.ray_oris[idx]
        ray_dirs = self.ray_dirs[idx]
        pixels = self.pixels[idx]

        return ray_dirs, ray_oris, pixels

    def __len__(self):
        return self.num_images

def sample_rays(raydirs, rayoris, sample_z_values):
    rays = rayoris[..., None, :] + raydirs[..., None, :] * sample_z_values[..., :, None]
    return rays, sample_z_values

def sample_viewdirs(raydirs):
    view_dirs = raydirs / torch.linalg.norm(raydirs, dim=-1, keepdims=True)
    return view_dirs

def predict_to_rgb(sigma, rgb, z_vals, rays_d, white_bkgd):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
    dists = dists * torch.linalg.norm(rays_d[..., None, :], dim=-1)
    alpha = 1. - torch.exp(-sigma * dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])
    return rgb_map, depth_map, acc_map, weights

def sample_pdf(bins, weights, N_samples, det=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples

class NeRF(nn.Module):
    def __init__(self, x_posendim = 10, nwidth = 256, ndepth = 8, view_posendim = 4):
        super().__init__()
        xdim = (x_posendim*2 + 1) * 3 # 3 for RGB, 2 for sin/cos, all 63 dimensions
        layers = []
        layers_in = [nwidth] * ndepth
        layers_in[0] = xdim
        layers_in[5] = nwidth + xdim

        for i in range(ndepth):
            layers.append(nn.Linear(layers_in[i], nwidth))

        if view_posendim > 0:
            view_dim = (view_posendim * 2 + 1) * 3
            self.view_embedding = Embedder(view_posendim)
            self.head = ViewDependentHead(nwidth, view_dim)
        else:
            self.view_embedding = None
            self.head = NoViewDirHead(nwidth, 4)

        self.xembed = Embedder(x_posendim)
        self.layers = nn.Sequential(*layers)

    def forward(self, x, view_dirs):
        xshape = x.shape
        x = self.xembed(x)

        if self.view_embedding is not None:
            view_dirs = view_dirs[:, None].expand(xshape)
            view_dirs = self.view_embedding(view_dirs)
        raw_x = x
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x))
            if i == 4:
                x = torch.cat([x, raw_x], -1)

        return self.head(x, view_dirs)
