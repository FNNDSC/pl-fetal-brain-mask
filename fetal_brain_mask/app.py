# fetal_brain_mask ds ChRIS plugin app
#
# (c) 2021 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

from chrisapp.base import ChrisApp
from argparse import ArgumentDefaultsHelpFormatter
import colorlog
import logging
import os
from os import path
from glob import glob
import shutil
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import nibabel as nib
import numpy as np

from fetal_brain_mask.predict import MaskingTool

Gstr_title = """
______   _        _  ______           _        ___  ___          _    
|  ___| | |      | | | ___ \         (_)       |  \/  |         | |   
| |_ ___| |_ __ _| | | |_/ /_ __ __ _ _ _ __   | .  . | __ _ ___| | __
|  _/ _ \ __/ _` | | | ___ \ '__/ _` | | '_ \  | |\/| |/ _` / __| |/ /
| ||  __/ || (_| | | | |_/ / | | (_| | | | | | | |  | | (_| \__ \   < 
\_| \___|\__\__,_|_| \____/|_|  \__,_|_|_| |_| \_|  |_/\__,_|___/_|\_\\
"""


handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(thin_green)s[%(asctime)s]%(reset)s  '
    '%(log_color)s%(levelname)-8s'
    '%(cyan)s%(filename)s:%(funcName)s:%(lineno)-4s | '
    '%(log_color)s%(message)s%(reset)s'
))

logger = colorlog.getLogger()
logger.addHandler(handler)

class Fetal_brain_mask(ChrisApp):
    """
    A ChRIS app for automatic image masking of fetal brain MRI.
    """
    PACKAGE                 = __package__
    TITLE                   = 'Automatic masking of fetal brain images using deep learning'
    CATEGORY                = 'Segmentation'
    TYPE                    = 'ds'
    ICON                    = '' # url of an icon image
    MAX_NUMBER_OF_WORKERS   = 60 # Override with integer value
    MIN_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MAX_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT           = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT           = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def __init__(self):
        super().__init__()
        self.formatter_class = ArgumentDefaultsHelpFormatter

    def define_parameters(self):
        # this is a future spec
        # https://github.com/FNNDSC/chrisapp/issues/6
        self.add_argument(
            '-p', '--inputPathFilter',
            dest='inputPathFilter',
            help='selection (glob) for which files to evaluate',
            default='*.nii',
            type=str,
            optional=True
        )
        self.add_argument(
            '--overlay-background',
            dest='overlay_background',
            help='multiplier for values of masked voxels, '
                 'given as a comma-separated list of numbers [0, 1). '
                 "'0.0' effectively produces the extracted brain. "
                 "A low, non-zero value such as '0.2' produces volumes which "
                 "are helpful for visualization.",
            default='0.0,0.2',
            type=str,
            optional=True
        )
        self.add_argument(
            '-s', '--no-smooth',
            type=bool,
            optional=True,
            default=False,
            dest='no_smooth',
            help='skip post-processing on images including morphological closing and defragmentation'
        )
        self.add_argument(
            '-l', '--skipped-list',
            type=str,
            optional=True,
            default='mask_skipped.txt',
            dest='skipped_list',
            help='produce an output file containing the names of files which were skipped'
        )

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        self.set_verbosity(options.verbosity)

        input_files = glob(path.join(options.inputdir, options.inputPathFilter))
        mask_folder = path.join(options.outputdir, 'mask')
        os.mkdir(mask_folder)

        overlay_folder = path.join(options.outputdir, 'extracted')
        bg_mult = None
        if options.overlay_background:
            os.mkdir(overlay_folder)
            bg_mult = [float(s.strip()) for s in options.overlay_background.split(',')]
            for mult in bg_mult:
                os.mkdir(path.join(overlay_folder, str(mult)))

        masker = MaskingTool()

        def process_nofail(input_filename: str) -> bool:
            try:
                logger.info('Processing ' + input_filename)
                input_basename = path.basename(input_filename)

                src_vol = nib.load(input_filename)
                src_data = src_vol.get_fdata(caching='unchanged')
                mask_data = masker.mask_tensor(src_data, not options.no_smooth)

                def save(data, outputdir, suffix):
                    output_basename = self.change_nii_extension(input_basename, suffix)
                    output_filename = path.join(outputdir, output_basename)
                    # create Nifti object with same header, but new data
                    out_vol = src_vol.__class__(data, src_vol.affine, header=src_vol.header)
                    nib.save(out_vol, output_filename)

                save(mask_data, mask_folder, '_mask.nii')

                for mult in bg_mult:
                    over_data = np.clip(mask_data, mult, 1.0)
                    over_data *= src_data
                    save(over_data, path.join(overlay_folder, str(mult)), '_brain.nii')

                return True
            except Exception as e:
                logger.error(e)
                logger.error('Failed to create a mask for ' + input_filename)
                return False

        # if we develop a GPU version, then change max_workers to GPU count
        with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
            successes = pool.map(lambda t: process_nofail(t), input_files)

        if options.skipped_list:
            skipped = [fname for fname, success in zip(input_files, successes) if not success]
            skipped_text = '\n'.join(skipped)
            with open(path.join(options.outputdir, options.skipped_list), 'w') as f:
                f.write(skipped_text)

    def show_man_page(self):
        """
        Print the app's man page.
        """
        self.print_help()

    @classmethod
    def set_verbosity(cls, verbosity):
        # legacy thing in chrisapp==2.1.0
        verbosity = int(verbosity)
        if verbosity > 0:
            print(Gstr_title, flush=True)
            print('Version: %s' % cls.get_version(), flush=True)
            logger.setLevel(logging.DEBUG if verbosity > 1 else logging.INFO)

        tf.get_logger().setLevel(logging.getLevelName(logger.level))

    @staticmethod
    def change_nii_extension(filename: str, suffix):
        if '.nii' not in filename:
            logger.warning(filename + ' does not contain .nii file extension')
            return filename + suffix
        if filename.endswith('.nii.gz'):
            return filename[:-7] + suffix
        return filename[:-4] + suffix
