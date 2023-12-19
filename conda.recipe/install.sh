# to install sigpy from scratch:

git clone https://github.com/ZhengguoTan/sigpy.git sigpy_github

cd sigpy_github

conda create -n sigpy_github python=3.10
conda activate sigpy_github

conda install -c anaconda pip

python -m pip install torch torchvision torchaudio

python -m pip install tqdm
python -m pip install numba
python -m pip install scipy
python -m pip install pywavelets
python -m pip install h5py
python -m pip install matplotlib

# please -
# (1) check your cuda version here
# (2) log into gpu machine when you are in hpc
# https://docs.cupy.dev/en/stable/install.html
pip install cupy-cuda11x

pip install -e .