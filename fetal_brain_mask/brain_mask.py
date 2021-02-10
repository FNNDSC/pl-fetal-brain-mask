#Brain masking tool
#Developed by Alejandro Valdes

import argparse
import os
import cv2
import sys
import glob
import numpy as np
from tqdm import tqdm
from medpy.io import load, save
from models.model import Unet
from skimage.transform import resize
from skimage.measure import label
from skimage.morphology import binary_closing, cube

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--target-dir',
    nargs='?',
    required=True,
    help='path to the dir that contains images, it will recursivelly look for all .nii images and provide a mask for them, defaults to the images directory in the project.')

parser.add_argument('--remasking',
    dest='remasking',
    action='store_true',
    help='flag to indicate already masked images should be re masked, rewritting of all *_mask.nii found, defaults to False')
parser.add_argument('--no-remasking',
    dest='remasking',
    action='store_false',
    help='flag to indicate the skipping of already masked images, if there is a file of the same name with _mask, it will be skipped')
parser.set_defaults(remasking=False)

parser.add_argument('--post-processing',
    dest='post_processing',
    action='store_true',
    help='flag to indicate predicted mask should be post processed (morphological closing and defragged), defaults to True')
parser.add_argument('--no-post-processing',
    dest='post_processing',
    action='store_false',
    help='flag to indicate predicted mask should not be post processed (morphological closing and defragged)')
parser.set_defaults(post_processing=True)

parser.add_argument('--match',
    nargs='+',
    help='Specify if only files with certain words should be masked, not case sensitive')

model_type = 'unet'

args = parser.parse_args()
target_dir = args.target_dir
remasking = args.remasking
post_processing = args.post_processing
match = args.match

if match:
    for i in range(len(match)):
        match[i] = match[i].lower()

def __normalize0_255(img_slice):
    '''Normalizes the image to be in the range of 0-255
    it round up negative values to 0 and caps the top values at the
    97% value as to avoid outliers'''
    img_slice[img_slice < 0] = 0
    flat_sorted = np.sort(img_slice.flatten())

    #dont consider values greater than 97% of the values
    top_3_limit = int(len(flat_sorted) * 0.97)
    limit = flat_sorted[top_3_limit]

    img_slice[img_slice > limit] = limit

    rows, cols = img_slice.shape
    #create new empty image
    new_img = np.zeros((rows, cols))
    max_val = np.max(img_slice)
    if max_val == 0:
        return new_img

    #normalize all values
    for i in range(rows):
        for j in range(cols):
            new_img[i,j] = int((
                float(img_slice[i,j])/float(max_val)) * 255)

    return new_img

def __resizeData(image, target=(256, 256)):
    image = np.squeeze(image)
    resized_img = []
    for i in range(image.shape[0]):
        img_slice = cv2.resize(image[i,:,:], target)
        resized_img.append(img_slice)

    image = np.array(resized_img, dtype=np.uint16)

    return image[..., np.newaxis]

def __postProcessing(mask):
    pred_mask = binary_closing(np.squeeze(mask), cube(2))

    try:
        labels = label(pred_mask)
        pred_mask = (labels == np.argmax(np.bincount(labels.flat)[1:])+1).astype(np.uint16)
    except:
        pred_mask = pred_mask

    return pred_mask


def getImageData(fname):
    '''Returns the image data, image matrix and header of
    a particular file'''
    data, hdr = load(fname)
    # axes have to be switched from (256,256,x) to (x,256,256)
    data = np.moveaxis(data, -1, 0)

    norm_data = []
    # normalize each image slice
    for i in range(data.shape[0]):
        img_slice = data[i,:,:]
        norm_data.append(__normalize0_255(img_slice))

    # remake 3D representation of the image
    data = np.array(norm_data, dtype=np.uint16)

    data = data[..., np.newaxis]
    return data, hdr

def main():
    # get all files in target dir that end with nii
    all_files = glob.glob(target_dir+'/**/*.nii', recursive=True)
    all_gz_files = glob.glob(target_dir+'/**/*.nii.gz', recursive=True)

    all_files += all_gz_files

    if match:
        all_files = [f for f in all_files if any(m in f.lower() for m in match)]

    # ignore masks
    files = [f for f in all_files if '_mask.nii' not in f]
    masks = [f for f in all_files if f not in files]

    if not remasking:
        files = [f for f in files if f[:-4] + '_mask.nii' not in masks]

    print('Found %d NIFTI files'%len(files))

    if remasking:
        print('Remasking set to True, remasking all images found')
    else:
        print('Remasking set to False, masking only images without a [file name]_mask.nii file')

    if post_processing:
        print('Post processing set to True, post processing output masks')
    else:
        print('Post processing set to False, not post processing output masks')

    # ignore masks
    files = [f for f in files if '_mask.nii' not in f]

    if len(files) == 0:
        print('No NIFTI files found, exiting')
        sys.exit(0)

    if model_type == 'unet':
        print('Loading Unet model')
        model = Unet()

    skipped = []

    for img_path in tqdm(files):

        try:
            img, hdr = getImageData(img_path)
            resizeNeeded = False

            if model_type == 'unet':
                if img.shape[1] != 256 or img.shape[2] != 256:
                    original_shape = (img.shape[2], img.shape[1])
                    img = __resizeData(img)
                    resizeNeeded = True


            res = model.predict_mask(img)

            if post_processing:
                res = __postProcessing(res)

            if resizeNeeded:
                res = __resizeData(res.astype(np.uint16), target=original_shape)

            # remove extra dimension
            res = np.squeeze(res)

            # return result into shape (256,256, X)
            res = np.moveaxis(res, 0, -1)

            # Save result
            img_path = img_path[:img_path.rfind('.')]

            # this is for files ending in .nii.gz
            if '.nii' in img_path:
                img_path = img_path[:img_path.rfind('.')]

            save(res, img_path + '_mask.nii', hdr)

        except Exception as e:
            print(e)
            print('not stopping')
            skipped.append(img_path)
            continue

    if len(skipped) > 0:
        print("%d images skipped."%len(skipped))
        skipped_file = open('skipped.txt', 'w+')
        for img_path in skipped:
            skipped_file.write(img_path+'\n')
            print(img_path)
        skipped_file.close()

if __name__ == '__main__':
    main()
