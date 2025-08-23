import time
import os
import numpy as np
import h5py
import json
from imageio import imread
from cv2 import resize as imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.
    """
    assert dataset in {'SECOND_CC'}  # Only these datasets are supported
    os.mkdir(output_folder)
    # Read Karpathy JSON annotation file
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Containers for image paths and captions (train/val/test)
    train_image_paths = []
    train_image_captions = []

    val_image_paths = []
    val_image_captions = []

    test_image_paths = []
    test_image_captions = []

    # Count word frequencies
    word_freq = Counter()
    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency with tokens
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:  # Discard captions longer than max_len
                captions.append(c['tokens'])

        if len(captions) == 0:  # Skip image if no valid captions
            continue

        # Build paths depending on dataset
        if dataset == 'SECOND_CC':
            # For SECOND_CC we have four inputs per sample:
            #  - RGB "A" (pre-change)
            #  - RGB "B" (post-change)
            #  - Semantic "A" (semantic mask aligned with A)
            #  - Semantic "B" (semantic mask aligned with B)
            path1 = os.path.join(image_folder, img['split'], 'rgb', 'A', img['filename'])
            path2 = os.path.join(image_folder, img['split'], 'rgb', 'B', img['filename'])
            path_semantic1 = os.path.join(image_folder, img['split'], 'sem', 'A', img['filename'])
            path_semantic2 = os.path.join(image_folder, img['split'], 'sem', 'B', img['filename'])
            path = [path1, path2, path_semantic1, path_semantic2]

        # Split images into train/val/test
        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity checks
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Build word map (dictionary)
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Define base name for output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map as JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Fix random seed for reproducibility
    seed(123)

    # Process each split (currently only TEST is enabled here)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Store number of captions per image as metadata
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset for images
            if dataset == 'SECOND_CC':
                # Shape explanation:
                #   images.shape = (N, 4, 3, 256, 256)
                #   4 => four views per sample: [RGB_A, RGB_B, SEM_A, SEM_B]
                #   3 => channels (converted to 3 if grayscale)
                #   256x256 => spatial size after resizing
                images = h.create_dataset('images', (len(impaths), 4, 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions for this image
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images depending on dataset
                if dataset == 'SECOND_CC':
                    # --- RGB inputs (A and B) ---
                    img_A = imread(impaths[i][0])
                    img_B = imread(impaths[i][1])

                    # If grayscale, expand to 3 channels
                    if len(img_A.shape) == 2:
                        img_A = np.stack([img_A] * 3, axis=2)
                    if len(img_B.shape) == 2:
                        img_B = np.stack([img_B] * 3, axis=2)

                    # Resize to 256x256 and set channel-first (C,H,W)
                    img_A = imresize(img_A, (256, 256)).transpose(2, 0, 1)
                    img_B = imresize(img_B, (256, 256)).transpose(2, 0, 1)
                    assert img_A.shape == (3, 256, 256)
                    assert img_B.shape == (3, 256, 256)
                    assert np.max(img_A) <= 255
                    assert np.max(img_B) <= 255

                    # --- Semantic inputs (A and B) ---
                    # Note: we treat white (255,255,255) as "void" and map it to black (0,0,0)
                    img_semantic_A = imread(impaths[i][2])
                    mask = np.all(img_semantic_A == [255, 255, 255], axis=-1)
                    img_semantic_A[mask] = [0, 0, 0]

                    img_semantic_B = imread(impaths[i][3])
                    mask = np.all(img_semantic_B == [255, 255, 255], axis=-1)
                    img_semantic_B[mask] = [0, 0, 0]

                    # Ensure semantic maps are 3-channel
                    if len(img_semantic_A.shape) == 2:
                        img_semantic_A = np.stack([img_semantic_A] * 3, axis=2)
                    if len(img_semantic_B.shape) == 2:
                        img_semantic_B = np.stack([img_semantic_B] * 3, axis=2)

                    # Resize and channel-first
                    img_semantic_A = imresize(img_semantic_A, (256, 256)).transpose(2, 0, 1)
                    img_semantic_B = imresize(img_semantic_B, (256, 256)).transpose(2, 0, 1)
                    assert img_semantic_A.shape == (3, 256, 256)
                    assert img_semantic_B.shape == (3, 256, 256)
                    assert np.max(img_semantic_A) <= 255
                    assert np.max(img_semantic_B) <= 255

                    # Save in the fixed order: [RGB_A, RGB_B, SEM_A, SEM_B]
                    # This fixed ordering is the reason for the leading dimension being 4.
                    images[i] = [img_A, img_B, img_semantic_A, img_semantic_B]

                else:
                    # For datasets with a single image per sample
                    img = imread(impaths[i])
                    if len(img.shape) == 2:
                        img = np.stack([img] * 3, axis=2)
                    img = imresize(img, (256, 256)).transpose(2, 0, 1)
                    assert img.shape == (3, 256, 256)
                    assert np.max(img) <= 255
                    images[i] = img

                # Encode captions
                for j, c in enumerate(captions):
                    enc_c = [word_map['<start>']] + \
                            [word_map.get(word, word_map['<unk>']) for word in c] + \
                            [word_map['<end>']] + \
                            [word_map['<pad>']] * (max_len - len(c))

                    # Caption length (including <start> and <end>)
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Final sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and lengths
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


if __name__ == '__main__':

    print('creating files START at: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
    create_input_files(dataset='SECOND_CC',
                       karpathy_json_path=r"D:\SECOND-CC-AUG\SECOND-CC-AUG.json",
                       image_folder=r"D:\SECOND-CC-AUG",
                       captions_per_image=5,
                       min_word_freq=10,
                       output_folder=r'C:\Users\AliCanKaraca\Desktop\DL\Share\createdFileBlackAUG',
                       max_len=50)

    print('create_input_files END at: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
