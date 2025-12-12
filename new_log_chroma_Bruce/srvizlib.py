"""
Bruce A. Maxwell
Fall 2025

A set of useful classes for reading and applying an image to an SR-estimation network.

- Basic data set loader for loading and caching 16-bit TIFF images given as a list of image paths (e.g. from the command line)
- Image transforms
  - ToTensor
  - CenterCropToSize
  - ToLogRGB

Images are normalized to [0, 1] by the Dataset before applying any transforms.

Additional modifications should come from image transforms passed to the Dataset

Requires

openCV (cv2)

Numpy  (numpy)
tifffile
tqdm
torch
torchvision

"""

import os
import sys
import cv2
import numpy as np
#import tifffile
#from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ToTensor(object):
    """
    Convert NumPy ndarrays or PyTorch tensors in a sample to PyTorch tensors.

    Converts:
    - 'image': (H, W, C) -> (C, H, W)
    """
    def __call__(self, sample):
        if isinstance(sample['image'], np.ndarray ):
            sample['image'] = torch.from_numpy(sample['image'])

        # Format image to (C, H, W)
        if sample['image'].ndim == 3 and sample['image'].shape[-1] in {1, 3, 4}:
            sample['image'] = sample['image'].permute(2, 0, 1)

        return sample


class ToLogRGB(torch.nn.Module):
    """Convert 16-bit linear RGB image to log RGB."""

    def __init__(self):
        super().__init__()
    
    def forward(self, sample):
        augmented_image = sample['image'].float() * 65535.0
        zero_mask = (augmented_image == 0)
        log_img = torch.log(augmented_image)
        log_img[zero_mask] = 0
        log_max = torch.log(torch.tensor(65535.0, device=augmented_image.device))
        log_img_normalized = torch.clamp(log_img / log_max, 0, 1)
        sample['image'] = log_img_normalized
        return sample    

class CenterCropToSize(torch.nn.Module):
    """
    Center-crops image and associated spatial maps to a fixed size on the GPU.

    Args:
        target_size (tuple): (height, width) to crop to.
    """
    def __init__(self, target_size=(512, 512)):
        super().__init__()
        self.target_size = target_size

    def forward(self, sample):
        h_crop, w_crop = self.target_size
        img = sample['image']
        h, w = img.shape[-2:]

        if h < h_crop or w < w_crop:
            raise ValueError(f"Input size ({h}, {w}) smaller than crop size {self.target_size}")

        top = (h - h_crop) // 2
        left = (w - w_crop) // 2

        def crop_tensor(tensor):
            return tensor[..., top:top + h_crop, left:left + w_crop]

        sample['image'] = crop_tensor(sample['image'])

        return sample


class BasicDataset_16BitTIFF(Dataset):

    def __init__(self, images, ds_device, image_transforms):

        # no augmentations, just load and pre-process the images for application to the network
        self.image_paths = images
        self.device = torch.device(ds_device) if ds_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_transforms = image_transforms

        self.image_cache = {}
        self._preload_images()

        # make a list of keys so we have a fixed index for each image
        self.image_list = list(self.image_cache.keys())
        print(self.image_list)

    def _preload_images(self, max_workers=8):
        print("Preloading images and ISD/segment maps into memory...")

        for path in self.image_paths:
            self.load_and_cache( path )
        
        # Launch parallel threads
#        with ThreadPoolExecutor(max_workers=max_workers) as executor:
#            list(tqdm(executor.map(self.load_and_cache, self.image_paths), total=len(self.image_paths), desc="Loading", unit="img"))


    def load_and_cache(self, img_path):
        if img_path in self.image_cache:
            return
        try:
            # alternative code using OpenCV
            src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            src = cv2.cvtColor( src, cv2.COLOR_BGR2RGB )

            # use tifffile to read the data
            #src = tifffile.imread(img_path)
            self.image_cache[img_path] = { 'image':src, 'img_name':img_path }
            
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")

    def __len__(self):
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int or torch.Tensor): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'image': The processed image as a NumPy array.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        print("processing index %d" % (idx))

        # have to convert to a tensor before putting it on the device
        image = torch.from_numpy(self.image_cache[ self.image_list[idx] ]['image'])
        
        # map the image to the device
        image = image.to(self.device)

        # Normalize image (16-bit TIFF normalization)
        image = torch.clamp(image / 65535.0, min=0, max=1)

        # Create sample dictionary
        sample = {'image': image, 'img_name': self.image_list[idx] }

        # Apply additional image transformations (e.g., cropping, flipping)
        if self.image_transforms:
            sample = self.image_transforms(sample)

        assert not torch.any(torch.isnan(sample['image'])), "Image Tensor contains NaN values!"

        return sample


# src is assumed to be log RGB and in the range [0, 1] as provided by the DataLoader
def ProjectLogChrom( log_norm_src, isd_vector = None, isd_map = None, anchor = 10.7, rg_brightness = 400 ):

    assert log_norm_src.min() >= 0.0 and log_norm_src.max() <= 1.0, 'Expected values [0, 1] out of range'
    
    log_src = log_norm_src * 11.1
    log_shifted_src = log_src - anchor

    print("computing projection")
    if isd_map is not None: # use the map
        dot_product_map = np.einsum( 'ijk,ijk->ij', log_shifted_src, isd_map )
        dot_product_map = dot_product_map[:,:,np.newaxis]

        # get the vector to project the point onto the plane
        projection = dot_product_map * isd_map

    else: # use the vector
        dot_product_map = np.dot( log_shifted_src, isd_vector )
        dot_product_map = np.stack( [dot_product_map, dot_product_map, dot_product_map], axis=-1 )
        projection = dot_product_map * isd_vector

    print("computing log chrom")
    # project the points onto the log chrom plane
    log_project_src = log_src - projection

    ##### compute a color visualization
    
    # get the linear version of the src image
    lin_org_src = np.clip( np.exp( log_src ).astype(np.float32), 1, 65535 )

    # compute the rg chromaticity of the src image in linear space
    lin_norm_src = np.sum(lin_org_src, axis=2)
    r_chroma = lin_org_src[..., 0] / lin_norm_src
    g_chroma = lin_org_src[..., 1] / lin_norm_src
    b_chroma = lin_org_src[..., 2] / lin_norm_src
    lin_rg_src = np.clip( np.stack( [r_chroma, g_chroma, b_chroma], axis=-1 ) * rg_brightness * 256, 1, 65535 )

    # project the chroma image into log space
    log_rg_src = np.log( lin_rg_src )

    # compute the difference in log space
    diff_src = log_rg_src - log_project_src
    r_shift = np.mean(diff_src[:,:,0])
    g_shift = np.mean(diff_src[:,:,1])
    b_shift = np.mean(diff_src[:,:,2])
    print("r_shift, g_shift, b_shift: %.2f %.2f %.2f" % (r_shift, g_shift, b_shift) )

    # adjust the log projection by shifting
    log_project_cor_src = log_project_src + np.array( [r_shift, g_shift, b_shift] )
    #log_project_cor_src = log_project_src

    # get the linear version
    lin_project_src = np.exp( log_project_cor_src ).astype(np.float32)
    
    # convert to 8-bits
    lin_8bit_src = (np.clip( lin_project_src, 0, 65535 ) / 256.0).astype(np.uint8)
    lin_8bit_src = cv2.cvtColor( lin_8bit_src, cv2.COLOR_BGR2RGB )

    # return both the simple log chrom projection and the visualization
    return log_project_src, lin_8bit_src
    

# test function for the class
def main(argv):

    if len(argv) < 2:
        print("Usage: python3 %s <image file 1> ..." % (argv[0]) )
        return


    files = argv[1:]
    size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    transform_list = [                                          # Create the list of transformations
            ToTensor(),
            CenterCropToSize(target_size=(size, size)),
            ToLogRGB(),
        ]
    composed_transforms = transforms.Compose(transform_list)

    # create the data set
    dataset = BasicDataset_16BitTIFF( files, device, composed_transforms )
    print(dataset)

    # make a data loader
    dataloader = torch.utils.data.DataLoader( dataset )
    
    for item in dataloader:
        print("processed image:", item['img_name'] )

    return


if __name__ == "__main__":
    main(sys.argv)

    
