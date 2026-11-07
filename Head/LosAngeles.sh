#!/bin/bash
#JSUB -q gpu 
#JSUB -gpgpu "1"
#JSUB -m gpu05
#JSUB -e output/error.%J 
#JSUB -o output/output.%J 
#JSUB -J Head

# =======================
# 环境加载（按你集群实际情况调整）
# =======================
module load mpfr/4.1.0
module load gcc/11.3.0
module load cuda/11.8

export CUDA_HOME=/apps/software/cuda/11.8
export CC=/apps/software/gcc/gcc-11.3.0/bin/gcc
export CXX=/apps/software/gcc/gcc-11.3.0/bin/g++

source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate htd-mamba

export LD_LIBRARY_PATH=/apps/software/mpfr/mpfr-4.1.0/lib:/home/23031212296/.conda/envs/htd-mamba/lib:/home/23031212296/.conda/envs/htd-mamba/lib/python3.10/site-packages/torch/lib:/apps/software/gcc/11.3.0/lib64:$LD_LIBRARY_PATH

# =======================
# 运行：robust_v11（更偏向“目标突出 + 背景抑制稳定”的折中版）
# python CL_Main_LosAngeles.py --mode eval \
#   --config configs/abu_airport2_rankboost_v17_autopow.py \
#   --test_load_weight ckpt_160_.pt
# python CL_Main_LosAngeles.py --mode all --config configs/abu_airport2_abl_det_FULL_DSM.py
# DSS_ONLY
python CL_Main_LosAngeles.py --mode eval --config configs/abu_airport2_abl_det_DSS_ONLY.py --test_load_weight ./ckpt/abu-airport-2/FULL_DSM/ckpt_160_.pt

# DSS+CSS
python CL_Main_LosAngeles.py --mode eval --config configs/abu_airport2_abl_det_DSS_CSS.py --test_load_weight ./ckpt/abu-airport-2/FULL_DSM/ckpt_160_.pt

# DSS+ACS
python CL_Main_LosAngeles.py --mode eval --config configs/abu_airport2_abl_det_DSS_ACS.py --test_load_weight ./ckpt/abu-airport-2/FULL_DSM/ckpt_160_.pt

# FULL(DSM)（如果想只 eval，不再训练）
python CL_Main_LosAngeles.py --mode eval --config configs/abu_airport2_abl_det_FULL_DSM.py --test_load_weight ./ckpt/abu-airport-2/FULL_DSM/ckpt_160_.pt