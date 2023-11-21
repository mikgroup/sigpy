"""Diffusion Vector Sets (dvs)
for the use of customized diffusion directions on the scanner

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np
from pathlib import Path

list_CoordinateSystem = ['xyz', 'prs']
list_Normalisation = ['unity', 'maximum', 'none']

def _read_directions(line_str):

    line_sp1 = line_str.split('=')[1]
    line_sp2 = line_sp1.split(']')[0]

    return int(line_sp2)

def _read_from_list(line_str, list_str):

    line_sp1 = line_str.split('=')[1]

    for l in list_str:
        if l in line_sp1:
            return l

def _read_Vector(line_str):

    line_sp1 = line_str.split('[')[1]
    line_sp2 = line_sp1.split(']')[0]

    row_ind = int(line_sp2)

    line_sp1 = line_str.split('(')[1]
    line_sp2 = line_sp1.split(')')[0]
    line_vec = line_sp2.split(',')

    row_arr = [0 for n in range(len(line_vec))]

    for n in range(len(line_vec)):
        row_arr[n] = float(line_vec[n])

    return row_ind, row_arr

# %% read .dvs file
def read(dvs_file):
    FILE = Path(dvs_file)

    if not FILE.exists():
        print('ERROR: can not find ' + dvs_file)
        return None

    if not FILE.is_file():
        print('ERROR: ' + dvs_file + ' is not readable')
        return None

    file_content = FILE.read_text()
    file_lines = file_content.splitlines()


    list_Params = []
    list_Vector = []

    for line in file_lines:

        if line.startswith('Vector'):
            list_Vector.append(line)
        else:
            list_Params.append(line)


    for line in list_Params:

        if 'directions=' in line:

            directions = _read_directions(line)

        elif 'CoordinateSystem' in line:

            CoordinateSystem = _read_from_list(line, list_CoordinateSystem)

        elif 'Normalisation' in line:

            Normalisation = _read_from_list(line, list_Normalisation)


    Vector = np.zeros((directions, 3))

    for line in list_Vector:

        row_ind, row_arr = _read_Vector(line)

        Vector[row_ind, :] = row_arr


    return directions, CoordinateSystem, Normalisation, Vector


# %% write .dvs file
def write(file, Vector, CoordinateSystem='xyz', Normalisation='none'):

    num_Vector, num_dirs = Vector.shape

    with open(file, 'w') as fp:

        directions_str = '[directions=' + str(num_Vector) + ']'
        fp.write(directions_str + '\n')

        CoordinateSystem_str = 'CoordinateSystem = ' + CoordinateSystem
        fp.write(CoordinateSystem_str + '\n')

        Normalisation_str = 'Normalisation = ' + Normalisation
        fp.write(Normalisation_str + '\n')

        for n in range(num_Vector):

            Vector_str = 'Vector[' + str("%3d"%n) + '] = ( '

            for d in range(num_dirs):

                Vector_str = Vector_str + str("%9.6f" % Vector[n, d])

                if d+1 == num_dirs:
                    Vector_str = Vector_str + ' )'
                else:
                    Vector_str = Vector_str + ', '

            fp.write(Vector_str + '\n')

    print('finish writing to file ' + file)