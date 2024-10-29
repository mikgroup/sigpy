"""
Methods for DICOM loading, extraction and saving.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

# %%
import numpy as np
import os
import pydicom

import os.path
import sys
sys.path.insert(0, os.path.join(os.environ['BART_PATH'], 'python'))
import cfl

from optparse import OptionParser

# __all__ = ['extract_ep2d_diff']


def save_dicom_img(dcm_file, out_file, file_type='cfl'):
    """Save dicom file as image.

    Args:
        dcm_file (string): dicom file name (including the directory).
        file_type (string): which format to save ('cfl' or 'npy').
    """
    ds = pydicom.filereader.dcmread(dcm_file)

    img = ds.pixel_array

    print('> write to ' + out_file)

    if file_type == 'cfl':
        cfl.writecfl(out_file, img)
    elif file_type == 'npy':
        np.save(out_file, img)

    return None


def ep2d_diff(dcm_dir, start=1, incre=1,
              save=False, save_dir=os.getcwd(), save_file='test',
              save_type='cfl'):
    """Extract b values and g vectors from dicom headers.

    Args:
        dcm_dir (string): directories where dicom files are stored.
        start (int): starting number.
        incre (int): incremental number (step).

    Returns:
        b (array): b-value array.
        g (array): g-vectory array.
    """

    # initialization
    b = np.array([], dtype=np.float).reshape(0, 1)
    g = np.array([], dtype=np.float).reshape(0, 3)

    dcm_files = sorted(os.listdir(dcm_dir))
    stop = len(dcm_files)

    for index in np.arange(start=start, stop=stop+1, step=incre):
        dcm_file = dcm_files[index-1]
        # print(dcm_file)
        dcm_dir_file = os.path.join(dcm_dir, dcm_file)

        assert dcm_file.find(str(index).zfill(4)) >= 0
        assert os.path.isfile(dcm_dir_file) == True

        ds = pydicom.filereader.dcmread(dcm_dir_file)

        if save:
            img = ds.pixel_array
            if index == start:
                out = np.reshape(img, list(img.shape) + [1])
                print(out.shape)
            else:
                out = np.concatenate((out,
                        np.reshape(img, list(img.shape) + [1])),
                        axis=2)
            # save_dicom_img(dcm_dir_file, dcm_dir_file)

        # b values (b)
        element = ds[0x0019, 0x100C]
        assert element.name.find('B_value') >= 0
        bval = float(element.value)
        b = np.vstack([b, bval])

        # diffusion gradient directions (g)
        # for b-value equals 0, the dicom tag for g does not exist!
        if bval == 0:
            g = np.vstack([g, [0., 0., 0.]])
        else:
            element = ds[0x0019, 0x100E]
            assert element.name.find('DiffusionGradientDirection') >= 0
            g = np.vstack([g, element.value])

    if save:
        save_name = save_dir + '/' + save_file
        if save_type == 'cfl':
            cfl.writecfl(save_name, out)
        elif save_type == 'npy':
            np.save(save_name, out)

    return b, g


# %% 
if __name__ == "__main__":
    usage = "%prog [options] <INPUT dir> <OUTPUT b-values> <OUTPUT g>"

    parser = OptionParser(description="extract b-values (b) diffusion-gradient-directions (g) from dicom files", usage=usage)

    parser.add_option("--start", dest="file_start",
                      help="dicom file starting number", default=1)
    parser.add_option("--incre", dest="file_incre",
                      help="dicom file incremental number", default=1)

    (options, args) = parser.parse_args()

    file_start = int(options.file_start)
    file_incre = int(options.file_incre)

    dir = str(args[0])

    b, g = ep2d_diff(dir, start=file_start, incre=file_incre)

    np.save(str(args[1]), b)
    np.save(str(args[2]), g)
