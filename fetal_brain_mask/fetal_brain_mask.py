#
# fetal_brain_mask ds ChRIS plugin app
#
# (c) 2021 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

from chrisapp.base import ChrisApp


Gstr_title = """
  __     _        _   _               _                             _    _             
 / _|   | |      | | | |             (_)                           | |  (_)            
| |_ ___| |_ __ _| | | |__  _ __ __ _ _ _ __    _ __ ___   __ _ ___| | ___ _ __   __ _ 
|  _/ _ \ __/ _` | | | '_ \| '__/ _` | | '_ \  | '_ ` _ \ / _` / __| |/ / | '_ \ / _` |
| ||  __/ || (_| | | | |_) | | | (_| | | | | | | | | | | | (_| \__ \   <| | | | | (_| |
|_| \___|\__\__,_|_| |_.__/|_|  \__,_|_|_| |_| |_| |_| |_|\__,_|___/_|\_\_|_| |_|\__, |
                 ______                    ______                                 __/ |
                |______|                  |______|                               |___/ 
"""

class Fetal_brain_mask(ChrisApp):
    """
    A ChRIS app for automatic image masking of fetal brain MRI.
    """
    PACKAGE                 = __package__
    TITLE                   = 'Automatic masking of fetal brain images using deep learning'
    CATEGORY                = 'Segmentation'
    TYPE                    = 'ds'
    ICON                    = '' # url of an icon image
    MAX_NUMBER_OF_WORKERS   = 1  # Override with integer value
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

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        print(Gstr_title)
        print('Version: %s' % self.get_version())

    def show_man_page(self):
        """
        Print the app's man page.
        """
        self.print_help()
