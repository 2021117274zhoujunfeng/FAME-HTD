#!/bin/bash
#JSUB -q gpu 
#JSUB -gpgpu "1"
#JSUB -m gpu07
#JSUB -e output/error.%J 
#JSUB -o output/output.%J 
#JSUB -J FDAS3-Net
# 加载模块（调整顺序）
module load mpfr/4.1.0
module load gcc/11.3.0
module load cuda/11.8
# 设置 CUDA_HOME
export CUDA_HOME=/apps/software/cuda/11.8
# 设置编译器为 GCC 11.3.0
export CC=/apps/software/gcc/gcc-11.3.0/bin/gcc
export CXX=/apps/software/gcc/gcc-11.3.0/bin/g++
# 激活 Conda 环境
source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate htd-mamba
# 设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/software/mpfr/mpfr-4.1.0/lib:/home/23031212296/.conda/envs/htd-mamba/lib:/home/23031212296/.conda/envs/htd-mamba/lib/python3.10/site-packages/torch/lib:/apps/software/gcc/11.3.0/lib64:$LD_LIBRARY_PATH

# 进入工作目录
cd /home/23031212296/text/FDAS3-Net

# 运行 HTD-Mamba 测试脚本
# python CL_Main_LosAngeles.py
python run_ablation.py \
  --suite paper \
  --dataset abu-airport-2 \
  --path datasets/abu-airport-2.mat \
  --band 205 \
  --device cuda:0 \
  --epochs 100 \
  --no_eval
# python Sandiego2.py 




