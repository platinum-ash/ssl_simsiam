import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import xarray as xr
import numpy as np
import os

class SSL4EO_S12_Dataset(Dataset):
    def __init__(self, extracted_dir, transform=None):
        """
        Args:
          extracted_dir (str): Path to the directory containing multiple extracted .zarr folders
          transform (callable, optional): Transform to apply to each view
        """
        self.extracted_dir = extracted_dir
        self.zarr_folders = sorted([
            os.path.join(extracted_dir, d) for d in os.listdir(extracted_dir)
            if os.path.isdir(os.path.join(extracted_dir, d)) and d.endswith('.zarr')
        ])
        self.transform = transform

        # Prepare a list of (folder, sample_idx, time_idx)
        self.samples = []
        for folder in self.zarr_folders:
            # Open once to get sample/time dims - assumes consistent shape for all
            ds = xr.open_zarr(folder, consolidated=True)
            n_samples = ds.dims['sample']
            n_times = ds.dims['time']

            for sample_idx in range(n_samples):
                for time_idx in range(n_times):
                    self.samples.append((folder, sample_idx, time_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, sample_idx, time_idx = self.samples[idx]
        ds = xr.open_zarr(folder, consolidated=True)

        band_data = ds['bands']
        band_names = ds.coords['band'].values
        band_map = {name: i for i, name in enumerate(band_names)}

        r = band_data[sample_idx, time_idx, band_map['B04'], :, :].values
        g = band_data[sample_idx, time_idx, band_map['B03'], :, :].values
        b = band_data[sample_idx, time_idx, band_map['B02'], :, :].values

        rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
        rgb = np.stack([r, g, b], axis=-1).astype(np.float32)

        min_val = rgb.min()
        max_val = rgb.max()
        if max_val > min_val:
            rgb = (rgb - min_val) / (max_val - min_val)
        else:
            rgb = np.zeros_like(rgb)

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # CHW format

        if self.transform:
            view1, view2 = self.transform(rgb_tensor)
        else:
            view1 = view2 = rgb_tensor

        return view1, view2
