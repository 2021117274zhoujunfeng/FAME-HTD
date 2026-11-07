#!/bin/bash
#JSUB -q gpu
#JSUB -gpgpu "1"
#JSUB -m gpu05
#JSUB -e output/error.%J
#JSUB -o output/output.%J
#JSUB -J DBB-SSMNET

echo "Step 1: Loading modules"
module load mpfr/4.1.0
module load gcc/11.3.0
module load cuda/11.8

echo "Step 2: Setting environment variables"
export CUDA_HOME=/apps/software/cuda/11.8
export CC=/apps/software/gcc/gcc-11.3.0/bin/gcc
export CXX=/apps/software/gcc/gcc-11.3.0/bin/g++

echo "Step 3: Activating Conda environment"
source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate htd-mamba

echo "Step 4: Setting LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/apps/software/mpfr/mpfr-4.1.0/lib:/home/23031212296/.conda/envs/htd-mamba/lib:/home/23031212296/.conda/envs/htd-mamba/lib/python3.10/site-packages/torch/lib:/apps/software/gcc/11.3.0/lib64:$LD_LIBRARY_PATH

echo "Step 5: Changing to working directory"
# cd /home/23031212296/text/DBB-SSMNET

echo "Step 6: Running the script"
python run_sensitivity_losangeles.py --device cuda:0 --epoch 200


