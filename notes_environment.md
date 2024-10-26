On my windows PC, the python 3.7/3.8 conda cannot install jax properly, there are always dependencies missing.
Use python 3.9, it can install the jax sommothly with 'pip install jax jaxlib optax' only   


pip install torchvision
#conda install pytorch::torchvision

<!-- pip install patsy pyod xgboost -->
<!-- # conda install -c conda-forge pasty pyod xgboost -->
conda install -c conda-forge pyod xgboost
pip install patsy
pip install --upgrade --no-deps statsmodels
 
pip install easydict tqdm pandas seaborn scikit-learn dill dm-haiku
 
<!-- pip install protobuf==3.20[downgrade for jax]
pip install numpy==1.24.2[downgrade for jax] -->
 
 
pip install ucimlrepo

Interactive GPU session on Phoniex
# the highest level of cuda compatible is cuda 12.5
# module load cuda/12.1.1

module load cuda/11.8.0
 
# charged tier 
salloc -A gts-bzhao94 -N1 --mem-per-gpu=12G -qinferno -t0:15:00 --gres=gpu:V100:1

# free tier (for testing)
salloc -A gts-bzhao94 -N1 --mem-per-gpu=12G -qembers -t0:60:00 --gres=gpu:V100:1

# uninstall the 2.x dumpy first [numpy 2.x is not compatible with Jax veriosn we haves]
pip install numpy==1.24

python -c "import jax; print(jax.devices())"
[gpu(id=0)]

<!-- # downgrade to compatibe with tf 2.8

pip install protobuf==3.20.* -->

 

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Home directory: /storage/home/hcoda1/6/azhou60, with a 10 GB quota and a limit of 100,000 files/directories
Project storage: ~/p-bzhao94-0, for storing data. A quota of 1024GB is shared by all members of your research group that access this space.
Scratch directory: ~/scratch, with a 15 TB quota. Scratch is a temporary scratch space for jobs that require fast I/O and not for permanent data storage. All files older than 60 days are regularly deleted. It is NEVER backed up.


# Pace Verion Red Hat
Linux-5.14.0-427.26.1.el9_4.x86_64-x86_64-with-glibc2.34
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


=============How to set up cuda on PACE ===================
# free tier (for testing)
 
salloc -A gts-bzhao94 -N1 --mem-per-gpu=12G -qembers -t0:60:00 --gres=gpu:V100:1
module load cuda/11.8.0


Download Cudnn from Nvidia
choose the Local Installer for RedHat/CentOS 9.1 x86_64 (RPM) since your server is running a Red Hat-based Linux distribution (as indicated by the server version you shared earlier) and is on glibc2.34 which aligns with Red Hat/CentOS 9.x.


rpm2cpio cudnn-local-repo-rhel9-8.9.7.29-1.0-1.x86_64cuda11.rpm | cpio -idmv
~~~~~~~~~~
(py39tf28) [azhou60@atl1-1-03-006-35-0 ~]$ rpm2cpio cudnn-local-repo-rhel9-8.9.7.29-1.0-1.x86_64cuda11.rpm | cpio -idmv
./etc/yum.repos.d/cudnn-local-rhel9-8.9.7.29.repo
./var/cudnn-local-repo-rhel9-8.9.7.29
./var/cudnn-local-repo-rhel9-8.9.7.29/0315540C.pub
./var/cudnn-local-repo-rhel9-8.9.7.29/Local.md5
./var/cudnn-local-repo-rhel9-8.9.7.29/Local.md5.gpg
./var/cudnn-local-repo-rhel9-8.9.7.29/cudnn-local-0315540C-keyring.gpg
./var/cudnn-local-repo-rhel9-8.9.7.29/libcudnn8-8.9.7.29-1.cuda11.8.x86_64.rpm
~~~~~~~~~~~
rpm2cpio ./var/cudnn-local-repo-rhel9-8.9.7.29/libcudnn8-8.9.7.29-1.cuda11.8.x86_64.rpm | cpio -idmv


mkdir -p $HOME/cudnn/lib $HOME/cudnn/include
mkdir -p $HOME/cudnn/lib $HOME/cudnn/lib
cp -r ./usr/include/* $HOME/cudnn/include/
(py39tf28) [azhou60@atl1-1-03-006-35-0 ~]$ cp -r ./usr/lib64/* $HOME/cudnn/lib/
(py39tf28) [azhou60@atl1-1-03-006-35-0 ~]$ cp -r ./usr/include/* $HOME/cudnn/include/

(py39tf28) [azhou60@atl1-1-03-006-35-0 ~]$ export CPATH=$HOME/cudnn/include:$CPATH
(py39tf28) [azhou60@atl1-1-03-006-35-0 ~]$ export LD_LIBRARY_PATH=$HOME/cudnn/lib:$LD_LIBRARY_PATH
(py39tf28) [azhou60@atl1-1-03-006-35-0 ~]$ echo 'export CPATH=$HOME/cudnn/include:$CPATH' >> ~/.bashrc
(py39tf28) [azhou60@atl1-1-03-006-35-0 ~]$ echo 'export LD_LIBRARY_PATH=$HOME/cudnn/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
(py39tf28) [azhou60@atl1-1-03-006-35-0 ~]$ source ~/.bashrc

(py39tf28) [azhou60@atl1-1-03-006-35-0 NeuralLCB_C_phoniex]$ python test_gpus.py 
Num GPUs Available:  1
2024-10-22 22:21:22.604292: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-10-22 22:21:24.445014: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 31134 MB memory:  -> device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:3b:00.0, compute capability: 7.0
Matrix multiplication result shape: (10000, 10000)
Computation time: 4.0387256145477295 seconds
2.8.0

Install the jax GPU version:
pip install jaxlib==0.4.12+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax==0.4.12
pip install numpy==1.24 [downgrade for jax 0.4.12 gpu version]
pip install protobuf==3.20 [downgrade for tf2.8]
pip install easydict pandas tqdm joblib
pip install –-no-deps optax
testing 
python -c "import jax; print(jax.devices())"
python test_gpus.py
========================================================================= 

All of above does not work .
Try 
conda install jaxlib=*=*cuda* jax cuda-nvcc optax -c conda-forge -c nvidia
cuda 11.8 should use tf>=2.12 [PACE]
cuda 11.2 could use tf28 [ANNIEPC]

Not working!!


2024.Oct 24
module load cuda/12.1.1
conda create -n cuda12 python=3.9

https://jax.readthedocs.io/en/latest/installation.html
JAX supports NVIDIA GPUs that have SM version 5.2 (Maxwell) or newer. Note that Kepler-series GPUs are no longer supported by JAX since NVIDIA has dropped support for Kepler GPUs in its software.
You must first install the NVIDIA driver. You’re recommended to install the newest driver available from NVIDIA, but the driver version must be >= 525.60.13 for CUDA 12 on Linux.
pip installation: NVIDIA GPU (CUDA, installed via pip, easier)
There are two ways to install JAX with NVIDIA GPU support:
•	Using NVIDIA CUDA and cuDNN installed from pip wheels
•	Using a self-installed CUDA/cuDNN
The JAX team strongly recommends installing CUDA and cuDNN using the pip wheels, since it is much easier!
NVIDIA has released CUDA pip packages only for x86_64 and aarch64; on other platforms you must use a local installation of CUDA.
pip install --upgrade pip

# NVIDIA CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12]"
If JAX detects the wrong version of the NVIDIA CUDA libraries, there are several things you need to check:
•	Make sure that LD_LIBRARY_PATH is not set, since LD_LIBRARY_PATH can override the NVIDIA CUDA libraries.
•	Make sure that the NVIDIA CUDA libraries installed are those requested by JAX. Rerunning the installation command above should work.

pip install -U "jax[cuda12]"

https://www.tensorflow.org/install/pip#step-by-step_instructions
# For GPU users, this will install tf 2.17
pip install tensorflow[and-cuda]
# For CPU users
pip install tensorflow
# verify GPU setup
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import jax; print(jax.devices())"

There is warning from tf:
(cuda12) [azhou60@atl1-1-02-006-31-0 ~]$ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 2024-10-24 14:16:16.367408: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable TF_ENABLE_ONEDNN_OPTS=0. 2024-10-24 14:16:16.385740: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered 2024-10-24 14:16:16.406169: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered 2024-10-24 14:16:16.412382: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered 2024-10-24 14:16:16.428186: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations. To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags. 2024-10-24 14:16:19.757722: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

It looks like TensorFlow successfully recognized your GPU (/physical_device:GPU:0) based on the output of the tf.config.list_physical_devices('GPU') command. However, there are a few warnings related to factory registration attempts for cuFFT, cuDNN, and cuBLAS, as well as a missing TensorRT warning.
Here's a breakdown of the warnings:
1.	cuFFT, cuDNN, and cuBLAS registration warnings: These warnings indicate that there was an attempt to register the cuFFT, cuDNN, and cuBLAS factories, but these were already registered. This usually happens if there are multiple attempts to load or initialize these libraries. It shouldn't affect normal operations unless there are multiple conflicting versions of these libraries.
2.	TF-TRT Warning: Could not find TensorRT: TensorRT is an NVIDIA deep learning inference optimizer and runtime library. If you're not using TensorRT for optimizing TensorFlow models, you can ignore this warning. If you need TensorRT, you can install it separately.





Possible Solution:
Install tf cuda first, then install jax without cuda
pip install tensorflow[and-cuda]
pip install --upgrade pip
# Installs the wheel compatible with NVIDIA CUDA 12 and cuDNN 9.0 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_local]"

# Even if I run pip install tensorflow[and-cuda],  I still get the TensorRT warning. I will just let this warning go I think.







python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import jax; print(jax.devices())"