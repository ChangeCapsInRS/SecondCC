import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a DataLoader to create batches
    of (image, caption) pairs (and caption lengths).
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: directory where preprocessed files are stored
        :param data_name:   base name used for processed dataset files
        :param split:       one of {'TRAIN', 'VAL', 'TEST'}
        :param transform:   optional torchvision-style transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open the HDF5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder,
                                        self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Number of captions sampled per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load all encoded captions into memory
        with open(os.path.join(data_folder,
                               self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load all caption lengths into memory
        with open(os.path.join(data_folder,
                               self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Transformation pipeline for images (e.g. normalize, augmentations)
        self.transform = transform  # None by default

        # Total number of dataset items (equal to number of captions)
        # Each caption corresponds to one sample; same image can appear multiple times
        self.dataset_size = int(len(self.captions) / 1)

    def __getitem__(self, i):
        """
        Retrieve the i-th sample.
        Returns image, its corresponding caption, and caption length.
        For validation/testing also returns all captions of that image.
        """
        # The N-th caption corresponds to the (N // captions_per_image)-th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)

        # Apply transforms if provided
        if self.transform is not None:
            if img.shape == torch.Size([3, 256, 256]):
                # Single 3-channel image
                img = self.transform(img)
            elif img.shape == torch.Size([2, 3, 256, 256]):
                # If stored as multiple views (example case: 2 images stacked)
                ori_img = img  # (kept as-is though unused)
                img[0] = self.transform(img[0])
                img[1] = self.transform(img[1])

        # Get the i-th caption and its length
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])

        # Training split: return one caption-image pair
        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # Validation/Test: return image, caption, caplen + all captions for BLEU evaluation
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        """Return total number of samples in the dataset."""
        return self.dataset_size
