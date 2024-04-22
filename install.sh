conda remove -n gart --all -y
conda create -n gart python=3.9 -y

source activate gart

which python
which pip

# conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.1 -c pytorch -c nvidia -y
# FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git"
conda install pytorch3d -c pytorch3d -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

pip install lib_render/simple-knn
pip install lib_render/diff_gaussian_rasterization-alphadep
# git clone https://github.com/bytedance/MVDream lib_guidance/mvdream/extern/MVDream
pip install -e lib_guidance/mvdream/extern/MVDream

# cd lib_marchingcubes
# python setup.py build_ext --inplace
# python setup_c.py build_ext --inplace # for kdtree in cuda 11

pip install lpips
# pip install git+https://github.com/NVlabs/tiny-cuda-nn/@v1.6#subdirectory=bindings/torch # Not used in release version