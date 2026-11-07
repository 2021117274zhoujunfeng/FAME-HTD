# -*- coding: utf-8 -*-
"""CL_Main_LosAngeles.py

一个更“可直接运行/可复现”的入口脚本：
    - train   : 自监督训练（不使用目标端元）
    - select  : 依据 AUC4 从已保存 ckpt 中挑最优
    - eval    : 评估（注入目标端元，输出 AUC1~AUC5，保存 score/roc）
    - all     : train -> select -> eval

相较你上传的旧版，这里主要做了：
1) 修复 args.path / args.save_dir / args.log_dir / args.out_dir 实际不生效的问题；
2) 把“相似度门控阈值”统一改为**谱角单位 (radian)**（与 Data_Patch / CL_Train 保持一致）；
3) 增加 token_stride / token_activation 的配置项（默认更贴近 HTD-Mamba 官方实现的重叠分组）；
4) 增加 outlier repulsion（默认关）和 bg-only sinkhorn（默认关），按需启用。

运行示例：

(1) 训练+挑最优+评估
    python CL_Main_LosAngeles.py --mode all --path datasets/abu-airport-2.mat --device cuda:0

(2) 只评估某个 ckpt
    python CL_Main_LosAngeles.py --mode eval --path datasets/abu-airport-2.mat --device cuda:0 --test_load_weight ckpt_200_.pt

"""

from __future__ import annotations

import argparse


# ------------------------------------------------------------
# Python 版本兼容：argparse.BooleanOptionalAction (Python>=3.9)
# 有些集群环境仍然是 Python 3.8，会缺这个 Action，导致脚本在 parse_args() 处直接崩。
# 这里提供一个轻量 fallback：支持 --foo / --no-foo 两种写法（对齐 Python>=3.9 行为）。
# ------------------------------------------------------------
if not hasattr(argparse, "BooleanOptionalAction"):
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, required=False, help=None, **kwargs):
            # 自动补齐 --no-xxx 形式的反向开关
            opts = []
            for opt in option_strings:
                opts.append(opt)
                if opt.startswith("--") and (not opt.startswith("--no-")):
                    opts.append("--no-" + opt[2:])
            # 去重
            seen = set()
            uniq = []
            for o in opts:
                if o not in seen:
                    uniq.append(o)
                    seen.add(o)
            super().__init__(option_strings=uniq, dest=dest, nargs=0, default=default, required=required, help=help)

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string is None:
                setattr(namespace, self.dest, True)
                return
            if option_string.startswith("--no-"):
                setattr(namespace, self.dest, False)
            else:
                setattr(namespace, self.dest, True)

    argparse.BooleanOptionalAction = BooleanOptionalAction  # type: ignore[attr-defined]

import os
import scipy.io as sio
import json
import importlib.util
from typing import Dict, Any

from CL_Train import train, eval, select_best, sweep_eval


def load_user_config(config_path: str) -> Dict[str, Any]:
    """
    读取外部配置文件（减少脚本内大量 preset/默认值干扰，便于复现实验）。

    支持格式：
      - .py  : 需要提供 dict 变量 cfg / CONFIG / config 之一
      - .json: 标准 json dict

    约定：外部配置里的键名与 cfg 字段保持一致；若键名与 argparse 参数同名，会在 main() 中写回 args。
    """
    if not config_path:
        return {}
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")
    ext = os.path.splitext(config_path)[1].lower()
    if ext == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("json config 必须是 dict")
        return data
    if ext == ".py":
        spec = importlib.util.spec_from_file_location("user_cfg", config_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"无法加载配置文件: {config_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        for key in ("cfg", "CONFIG", "config"):
            if hasattr(mod, key):
                data = getattr(mod, key)
                if isinstance(data, dict):
                    return data
        raise ValueError("py config 需要提供 dict 变量 cfg / CONFIG / config 之一")
    raise ValueError("仅支持 .py / .json 配置文件")


def apply_train_preset(cfg: Dict, preset: str):
    p = str(preset).lower()

    if p == "paper":
        cfg["w_intra"] = 0.20
        cfg["w_ctx"] = 1.00
        cfg["ctx_thr_mode"] = "fixed"
        cfg["ctx_sim_thr"] = 0.32
        cfg["ctx_sim_tau"] = 1.0
        cfg["delta"] = 0.10
        cfg["delta_ctx"] = 0.10

        # 训练预设也把推理 matcher 关掉（避免 train_preset=paper 但 infer_preset=robust 时仍启用 extras）
        cfg["use_wavelet_match"] = False
        cfg["use_deriv_match"] = False

        cfg["use_augment"] = True
        cfg["aug_curriculum"] = True
        cfg["augment_view1_scale"] = 0.25
        cfg["augment_view2_scale"] = 0.65

    elif p == "converge":
        # 更“好收敛”的重训预设：先把 AUC1 拉稳，再在推理侧追 AUC5
        cfg["w_intra"] = 0.20
        cfg["w_ctx"] = 1.00
        cfg["delta"] = 0.10
        cfg["delta_ctx"] = 0.10

        cfg["patch_encode_sim"] = "cosine"
        cfg["patch_encode_sigma"] = 0.0

        # 增强：从弱到强（避免前期被扰动拖垮）
        cfg["augment_view1_scale_start"] = 0.0
        cfg["augment_view2_scale_start"] = 0.0
        cfg["aug_ramp_epochs"] = 120
        cfg["augment_view1_scale"] = 0.25
        cfg["augment_view2_scale"] = 0.65

        # drift：小一些更稳
        cfg["drift_p"] = 0.05
        cfg["drift_warmup"] = 60

        # 原型/几何：后期开（也可 CLI 覆盖）
        cfg["w_geom"] = 0.03
        cfg["w_proto"] = 0.03
        cfg["proto_start_epoch"] = 80
        cfg["proto_warmup"] = 80

    # robust: 不做额外修改（沿用 cfg 默认）
    pass


def apply_infer_preset(cfg: Dict, preset: str):
    p = str(preset).lower()

    if p == "paper":
        cfg["use_bg_osp"] = False
        cfg["use_wavelet_match"] = False
        cfg["use_deriv_match"] = False
        cfg["use_angle_gate"] = False
        cfg["score_mode"] = "htdmamba"
        cfg["score_post_norm"] = "none"

    elif p == "auc1":
        cfg["use_bg_osp"] = False
        cfg["use_wavelet_match"] = False
        cfg["use_deriv_match"] = False
        cfg["use_angle_gate"] = False
        cfg["score_mode"] = "htdmamba"
        cfg["score_post_norm"] = "none"

    elif p == "balanced":
        cfg["use_bg_osp"] = True
        cfg["bg_osp_alpha"] = 0.65
        cfg["bg_gate_mode"] = "sigmoid_sim"
        cfg["bg_gate_thr_q"] = 0.995

        cfg["angle_gate_w"] = 0.35
        cfg["angle_gate_k"] = 8.0  # 修复旧版 angle_gate_s 写错 key

        cfg["adaptive_fusion"] = True

        cfg["score_mode"] = "nbs_proto"
        cfg["use_target_aug"] = False
        cfg["target_aug_num"] = 0
        cfg["target_aug_scale"] = 0.0
        cfg["target_aug_noise"] = 0.0
        cfg["angle_gate_calib"] = True
        cfg["save_debug_for_best"] = True
        cfg["fusion_mode"] = "min_sup"
        cfg["fusion_floor"] = 0.02

    elif p == "balanced_v2":
        # 在 balanced 基础上：调大 NBS-T 的归一化 top_ratio，避免 AUC2 被 t_max 压扁
        cfg["use_bg_osp"] = True
        cfg["bg_osp_alpha"] = 0.65
        cfg["bg_gate_mode"] = "sigmoid_sim"
        cfg["bg_gate_thr_q"] = 0.995

        cfg["angle_gate_w"] = 0.35
        cfg["angle_gate_k"] = 8.0

        cfg["adaptive_fusion"] = True
        cfg["score_mode"] = "nbs_proto"

        cfg["nbs_t_norm"] = "topk_median"
        cfg["nbs_t_norm_mask"] = "angle_t0_gate"
        cfg["nbs_t_norm_top_ratio"] = 0.012

        cfg["use_target_aug"] = False
        cfg["target_aug_num"] = 0
        cfg["target_aug_scale"] = 0.0
        cfg["target_aug_noise"] = 0.0
        cfg["angle_gate_calib"] = True
        cfg["save_debug_for_best"] = True

    elif p in ["robust_v2", "fusion", "multi_evidence", "evidence"]:
        # ------------------------------------------------------------
        # New inference preset (v2): multi-evidence fusion
        # ------------------------------------------------------------
        # 目标：在保持 AUC1 排序能力的同时，显著抬高 AUC2（τ-Pd）——
        # 也就是让“真目标像素的分数”更靠近 1，而不是被多证据乘法抑制压到 0 附近。
        cfg["score_mode"] = "evidence_z"

        # target-bank pooling: use angle-aware weights + softmax-topk pooling
        # to reduce multiple-comparisons while keeping strong matches high.
        cfg["target_sim_pool"] = "softmax_topk"
        cfg["target_sim_topk"] = 5
        cfg["target_sim_temp"] = 0.05

        # wavelet / deriv match still useful as an extra evidence (not only suppression)
        cfg["use_wavelet_match"] = True
        cfg["use_deriv_match"] = True
        # derivative pooling defaults (reuse target pooling if not set)
        cfg["deriv_sim_pool"] = "softmax_topk"
        cfg["deriv_sim_topk"] = 5
        cfg["deriv_sim_temp"] = 0.05

        # keep background prototypes + (optional) OSP, but make it slightly milder
        cfg["use_bg_osp"] = True
        cfg["bg_osp_alpha"] = 0.55
        cfg["bg_osp_apply"] = "pixel"

        # evidence fusion weights (can be tuned)
        cfg["ev_w_sim_t"] = 0.9
        cfg["ev_w_margin"] = 1.4
        cfg["ev_w_wave"] = 0.5
        cfg["ev_w_deriv"] = 0.35
        cfg["ev_w_ang"] = 0.7
        cfg["ev_w_bg"] = 0.4
        cfg["ev_bias"] = 0.0
        cfg["score_temp"] = 1.0

        # soft gate: auto margin threshold + t0-angle gate for stability
        cfg["margin_thr"] = "auto_mad"
        cfg["margin_thr_k_mad"] = 2.2
        cfg["gate_k"] = 8.0
        cfg["ev_gate_angle_source"] = "t0"
        cfg["ev_gate_ang_pow"] = 1.0
        cfg["ev_gate_w"] = 1.0

        # post calibration (monotonic) to lift AUC2 without changing AUC1
        cfg["score_post_norm"] = "q"
        cfg["score_post_norm_q"] = 0.999
        cfg["score_post_norm_mask"] = "angle_t0_gate"
        cfg["score_post_norm_angle_t0_gate_thr"] = 0.5

        cfg["save_debug_for_best"] = True

    elif p in ["robust_v3", "core", "core_v1", "core_evidence"]:
        # robust_v3 (AUC1-first): Feature-space ACE + background OSP
        # 目标：优先拉升 AUC1（排序能力），同时用统计自适应抑制异质背景假警。
        cfg["infer_preset"] = "robust_v3"

        # --- core change: use ACE in feature space (test-time only) ---
        cfg["target_sim_metric"] = "ace"  # compute sim_t via ACE instead of cosine
        cfg["score_mode"] = "ace"  # final score = ACE(sim)

        # ACE covariance estimation (background is dominant)
        cfg["ace_bg_select"] = (
            "sim_bg_q"  # pick typical background by bg-prototype similarity
        )
        cfg["ace_bg_q"] = 0.50  # keep top-50% bg-typical pixels to estimate covariance
        cfg["ace_shrink"] = 0.05  # Ledoit-Wolf style shrink toward isotropic diag
        cfg["ace_ridge"] = 1e-3  # diagonal loading
        cfg["ace_eps"] = 1e-8

        # --- keep inference strictly test-time (no target in training) ---
        # to reduce multiple-comparisons false alarms, disable target-bank augmentation in robust_v3
        cfg["use_target_aug"] = False
        cfg["target_aug_num"] = 0

        # --- background orthogonal projection (prototype subspace) ---
        cfg["use_bg_osp"] = True
        cfg["bg_osp_k"] = 4
        cfg["bg_osp_alpha"] = 0.75
        cfg["bg_osp_apply"] = "both"
        cfg["bg_osp_method"] = "exact"
        cfg["bg_osp_ridge"] = 1e-3

        # --- disable extra gates that may reorder scores and hurt AUC1 ---
        cfg["use_angle_gate"] = False
        cfg["angle_aware_pool"] = False

        # --- keep a simple monotonic post-processing (does not change AUC1) ---
        cfg["score_post_norm"] = "q"
        cfg["score_post_norm_q"] = 0.999
        cfg["score_post_norm_mask"] = "none"
        cfg["score_post_pow"] = 1.5

        # debug for analysis
        cfg["save_debug_best"] = True

    elif p in ["robust_v4", "mt_camf", "mtmf_camf"]:
        # ------------------------------------------------------------
        # robust_v4: MT-CAMF Head (test-time only)
        # ------------------------------------------------------------
        # 目标：在不引入训练期目标谱的前提下，提升异质背景下的排序稳定性：
        #   - 用“聚类/原型分簇 + 自适应背景统计”的思路缓解非同质背景
        #   - 用“Mixture-Tuned”思想减少高相似背景离群点
        #   - 用“target ⊕ background mix bank”覆盖亚像元/混合像素
        # 最终仍输出一个单通道 score（AUC1~AUC5 兼容原流程）

        cfg["infer_preset"] = "robust_v4"

        # --- core metric: MT-CAMF (in feature space) ---
        cfg["target_sim_metric"] = "mt_camf"
        cfg["score_mode"] = "ace"  # direct use sim_t_all in [0,1]

        # --- background statistics (clustered by bg prototypes) ---
        # bg selection for covariance: use typical background pixels (by sim_bg quantile)
        cfg["mt_camf_bg_select"] = "sim_bg_q"
        cfg["mt_camf_bg_q"] = 0.50

        cfg["mt_camf_shrink"] = 0.05
        cfg["mt_camf_ridge"] = 1e-3
        cfg["mt_camf_min_samples"] = 64
        cfg["mt_camf_alpha_pow"] = 0.5
        cfg["mt_camf_alpha_clip_max"] = 1.5
        cfg["mt_camf_infeas_lambda"] = 3.0
        cfg["mt_camf_infeas_gamma"] = 4.0

        # --- target mix bank (scene-adaptive, eval only) ---
        cfg["target_aug_num"] = (
            0  # keep explicit aug off; mix bank already provides variability
        )

        cfg["mixbg_enable"] = True
        cfg["mixbg_samples"] = 8
        cfg["mixbg_alphas"] = [0.10, 0.20, 0.30]
        cfg["mixbg_angle_q"] = 0.70
        cfg["mixbg_pool_size"] = 4096
        cfg["mixbg_seed_offset"] = 2026
        cfg["mixbg_use_alpha_prior"] = True
        cfg["mixbg_alpha_sigma"] = 0.35

        # bank weighting (reduce multiple-comparisons)
        cfg["target_bank_weighting"] = "angle"
        cfg["target_bank_weight_sigma"] = 0.25
        cfg["target_bank_angle_clip"] = 0.60
        cfg["target_bank_min_weight"] = 1e-4

        # pool over bank (soft top-k)
        cfg["target_sim_pool"] = "softmax_topk"
        cfg["target_sim_topk"] = 5
        cfg["target_sim_temp"] = 0.07

        # --- keep gates OFF to avoid killing true target pixels (AUC1-sensitive) ---
        cfg["use_angle_gate"] = False
        cfg["angle_aware_pool"] = False

        # --- do NOT use OSP by default in robust_v4 (avoid over-subtracting target) ---
        cfg["use_bg_osp"] = False

        # --- simple monotonic post-processing (does not change AUC1 ranking) ---
        cfg["score_post_norm"] = "q"
        cfg["score_post_norm_q"] = 0.999
        cfg["score_post_norm_mask"] = "none"
        cfg["score_post_pow"] = 1.25

        cfg["save_debug_best"] = True

    elif p in ["robust_v5", "mix_osp_nbs", "consensus"]:
        # ------------------------------------------------------------
        # robust_v5: Consensus-BOP NBS Head (test-time only)
        # ------------------------------------------------------------
        # 面向你最新日志暴露的问题：
        #   - robust_v4/ACE/MT-CAMF 系列往往能把背景压得很干净（AUC3 很低），
        #     但会把一部分“混合/弱目标像素”也压下去，从而 AUC1 掉点。
        # 这里把“排序能力强的 NBS-Proto”作为主干，再叠加：
        #   1) 仅对像素侧做部分背景子空间剔除（BOP/OSP），减少结构性背景假警；
        #   2) endmember-bundle + mix-bank 覆盖端元变异/亚像元混合，减少 FN 尾部；
        #   3) 小波/导数作为“谱形证据”参与 NBS（抑制条纹/局部频扰）；
        #   4) 只用 angle-aware pooling 做 bank 逐像素软权重，不用硬角门控，避免误伤。
        # 训练阶段仍不注入目标谱（全部发生在 eval 阶段）。

        cfg["infer_preset"] = "robust_v5"

        # --- keep NBS-Proto backbone (strong AUC1 ranking) ---
        cfg["score_mode"] = "nbs_proto"
        cfg["target_sim_metric"] = "cosine"

        # --- partial background orthogonal projection (pixel-side only) ---
        cfg["use_bg_osp"] = True
        cfg["bg_osp_k"] = 4
        cfg["bg_osp_alpha"] = 0.75
        cfg["bg_osp_apply"] = "pixel"  # avoid over-subtracting target bank
        cfg["bg_osp_method"] = "exact"
        cfg["bg_osp_ridge"] = 1e-3

        # --- target bank: pure aug + scene-adaptive mix bank (eval only) ---
        cfg["target_aug_num"] = 6

        cfg["mixbg_enable"] = True
        cfg["mixbg_samples"] = 6
        cfg["mixbg_alphas"] = [0.10, 0.20, 0.30, 0.40]
        cfg["mixbg_angle_q"] = 0.70
        cfg["mixbg_pool_size"] = 4096
        cfg["mixbg_seed_offset"] = 2027
        cfg["mixbg_use_alpha_prior"] = True
        cfg["mixbg_alpha_sigma"] = 0.20

        # bank weighting (reduce multiple-comparisons)
        cfg["target_bank_weighting"] = "angle"
        cfg["target_bank_weight_sigma"] = 0.18
        cfg["target_bank_angle_clip"] = 0.45
        cfg["target_bank_min_weight"] = 1e-4

        # pool over bank (soft top-k)
        cfg["target_sim_pool"] = "softmax_topk"
        cfg["target_sim_topk"] = 5
        cfg["target_sim_temp"] = 0.05

        # --- spectrum-shape evidence (wavelet + derivative) ---
        cfg["use_wavelet_match"] = True
        cfg["use_deriv_match"] = True
        cfg["wavelet_level_decay"] = 0.7
        cfg["deriv_sim_pool"] = "softmax_topk"
        cfg["deriv_sim_topk"] = 5
        cfg["deriv_sim_temp"] = 0.05

        # --- only use angle-aware pooling (no hard angle gate) ---
        cfg["use_angle_gate"] = False
        cfg["angle_aware_pool"] = True
        cfg["angle_pool_thr"] = 0.45
        cfg["angle_pool_k"] = 10.0
        cfg["angle_pool_min_w"] = 1e-4

        # --- NBS calibration: avoid background outliers inflating t_max/wave_max ---
        cfg["nbs_t_norm"] = "topk_median"
        cfg["nbs_t_norm_mask"] = "angle_t0_gate"
        cfg["nbs_t_norm_top_ratio"] = 0.005
        cfg["wave_sim_max"] = "topk_median"
        cfg["wave_sim_max_mask"] = "angle_t0_gate"

        # --- slightly stronger background suppression, protected by adaptive_fusion ---
        cfg["nbs_delta_bg"] = 0.08
        cfg["score_w_bg"] = 1.15
        cfg["score_w_wave"] = 0.65
        cfg["score_w_t"] = 1.0

        cfg["adaptive_fusion"] = True
        cfg["fusion_conf_thr"] = 0.70
        cfg["fusion_conf_k"] = 12.0
        cfg["fusion_min_sup"] = 0.15
        cfg["fusion_conf_use_angle"] = True
        cfg["fusion_conf_angle_source"] = "t0"
        cfg["fusion_conf_ang_coeff"] = 0.5
        cfg["fusion_conf_use_bg"] = True
        cfg["fusion_conf_bg_coeff"] = 0.5

        # --- monotonic post calibration (AUC1 unchanged, tau stability improved) ---
        cfg["score_post_norm"] = "q"
        cfg["score_post_norm_q"] = 0.999
        cfg["score_post_norm_mask"] = "angle_t0_gate"
        cfg["score_post_norm_angle_t0_gate_thr"] = 0.5
        cfg["score_post_pow"] = 1.0

        cfg["save_debug_best"] = True

    elif p == "robust_v6":
        # robust_v6: robust_v5 + spatial consensus refinement (test-time only)
        apply_infer_preset(cfg, "robust_v5")
        cfg["infer_preset"] = "robust_v6"

        # --- spatial refinement (fills holes, suppresses isolated peaks) ---
        cfg["spatial_refine_enable"] = True
        cfg["spatial_refine_mode"] = "neighbor_mean_blend"
        cfg["spatial_refine_ksize"] = 5
        cfg["spatial_refine_alpha"] = 0.60
        cfg["spatial_refine_iters"] = 1
        cfg["spatial_refine_pad"] = "reflect"
        cfg["spatial_refine_cuda"] = False

        # keep other robust_v5 settings
        cfg["save_debug_best"] = True

        return

    elif p == "robust_v7":
        # robust_v7: robust_v6 + Adaptive Sigmoid Calibration (ASC)
        # 说明：AUC1 只取决于排序，单调映射不会改变 AUC1；
        #      但 τ-Pd/τ-Pf 对绝对分数更敏感。
        #      我们在 score_post_norm 之后追加一个自适应 sigmoid（S-curve）标定，
        #      把背景的“中等分数”压向 0、把真目标分数推向 1，以同时提升 AUC2、压低 AUC3。
        apply_infer_preset(cfg, "robust_v6")
        cfg["infer_preset"] = "robust_v7"

        # --- ASC: sigmoid(score; mu, k) ---
        cfg["score_post_sigmoid"] = True
        cfg["score_post_sigmoid_mu"] = "q"  # mu 来自分位数
        cfg["score_post_sigmoid_q"] = 0.98  # 默认取 score 的 98% 分位作为拐点
        cfg["score_post_sigmoid_k"] = 20.0  # 越大越“硬阈值化”，通常 15~25 都可
        cfg["score_post_sigmoid_rescale"] = (
            True  # 端点锚定：强制 0→0、1→1，避免背景分数下限
        )
        cfg["score_post_sigmoid_mask"] = "none"  # 不建议 gate_low，会把 mu 估得过低
        cfg["score_post_sigmoid_min_keep"] = 1024

        # keep debug
        cfg["save_debug_best"] = True
        return

    elif p == "robust_v8":
        # ============================================================
        # robust_v8: 强化背景抑制（主攻 AUC3 ↓，同时尽量不伤 AUC1）
        # ============================================================
        # 你当前版本的问题集中在：少量结构性/稀有背景像素被打到“极高分”，
        # 造成 τ-Pf 曲线尾部抬起（AUC3 偏大）。
        # 这里给出一个“可直接替换”的推理侧 preset：
        #   1) 用 MT-CAMF(聚类背景统计 + Mixture-Tuned penalty) 替代纯 cosine
        #      ——专门针对异质背景 + outlier clutter 造成的假警；
        #   2) 用更保守的 target-bank pooling（topk_median）降低 multiple-comparisons 假警；
        #   3) 继承 robust_v7 的 ASC + 空间一致性细化，改善阈值稳定性。
        #
        # 经验上：robust_v8 往往能把“极少数高分背景”压下来，从而显著降低 AUC3。
        # ============================================================
        apply_infer_preset(cfg, "robust_v7")
        cfg["infer_preset"] = "robust_v8"

        # --- (1) Target similarity metric: MT-CAMF (test-time only) ---
        cfg["target_sim_metric"] = "mt_camf"
        # 选择哪些像素参与背景统计：用 sim_bg 的分位数挑“更像背景”的样本
        cfg["mt_camf_bg_select"] = "sim_bg_q"
        cfg["mt_camf_bg_q"] = 0.60
        # 协方差估计更稳健：shrink + ridge
        cfg["mt_camf_shrink"] = 0.05
        cfg["mt_camf_ridge"] = 1e-3
        cfg["mt_camf_min_samples"] = 96
        # mixture-tuned：abundance^pow * exp(-lambda * infeas)
        cfg["mt_camf_alpha_pow"] = 0.5
        cfg["mt_camf_alpha_clip_max"] = 1.5
        # 这两项越大，对“不可行的高投影背景(outlier clutter)”惩罚越强
        cfg["mt_camf_infeas_lambda"] = 4.0
        cfg["mt_camf_infeas_gamma"] = 6.0

        # --- (2) Bank-consensus pooling (reduce lucky single-hit false alarms) ---
        cfg["target_sim_pool"] = "topk_median"
        cfg["target_sim_topk"] = 5
        # 注：topk_median 不用 temp，但保留字段便于你后续回切 softmax_topk
        cfg["target_sim_temp"] = 0.07

        # wave/deriv 同样用更保守的 pooling（可选）
        cfg["wave_sim_pool"] = "topk_median"
        cfg["wave_sim_topk"] = 5
        cfg["deriv_sim_pool"] = "topk_median"
        cfg["deriv_sim_topk"] = 5

        # --- (3) Make mix-bank a bit more conservative (optional) ---
        # 若你的数据集“混合像素”不明显，可以把 mixbg_enable=False 获得更低假警。
        cfg["mixbg_enable"] = True
        cfg["mixbg_samples"] = 4
        cfg["mixbg_alphas"] = [0.1, 0.2, 0.3]
        cfg["mixbg_angle_q"] = 0.85
        cfg["mixbg_alpha_sigma"] = 0.25

        # keep debug
        cfg["save_debug_best"] = True
        return

    elif p == "robust_v9":
        # ============================================================
        # robust_v9: 在 robust_v8 基础上，进一步“只抑制不扩散”的空间后处理
        #            + 更强的背景 gate（更偏向 AUC3 ↓ 的版本）
        # ============================================================
        # 经验现象：neighbor_mean_blend 这种空间平滑虽然能降孤立噪点，
        # 但也容易把少数高分假警“扩散”成一团，导致 Pf 尾部抬起（AUC3 变大）。
        # robust_v9 的核心思想：
        #   1) 仍用 robust_v8 的 MT-CAMF + topk_median 降“幸运单击”假警；
        #   2) 空间侧改为 support_suppress（只会把分数压低，不会抬高），
        #      更适合你现在的“背景抑制”优化方向；
        #   3) 把 bg_gate 阈值从 0.90 分位下调，让更多背景像素参与抑制。
        # ============================================================
        apply_infer_preset(cfg, "robust_v8")
        cfg["infer_preset"] = "robust_v9"

        # --- (A) Spatial refinement: suppress-only (no score diffusion) ---
        cfg["spatial_refine_enable"] = True
        cfg["spatial_refine_mode"] = "support_suppress"  # 新增模式，见 CL_Train.py
        cfg["spatial_refine_ksize"] = 5
        cfg["spatial_refine_iters"] = 1
        cfg["spatial_refine_pad"] = "reflect"
        # 邻域均值阈值：越大越严格（假警↓，但可能伤细小目标）；可在 0.05~0.15 间扫
        cfg["spatial_refine_thr"] = 0.08
        # support = sigmoid(k*(nb-thr)) 的陡峭系数
        cfg["spatial_refine_k"] = 12.0
        # support^p（p>1 更“硬”）；一般 1.0~1.5
        cfg["spatial_refine_p"] = 1.0
        # 给边缘留一点最小支持度，避免把目标边界压得太狠
        cfg["spatial_refine_min_sup"] = 0.10

        # --- (B) Stronger background gate (helps suppress structured clutter) ---
        cfg["bg_gate_thr"] = "auto"
        cfg["bg_gate_thr_q"] = 0.75  # 原来 0.90 偏松；下调更利于 AUC3
        cfg["bg_gate_k"] = 32.0
        # conf 中背景因子的占比（增大可让 conf 更谨慎，从而 sup 更大、抑制更强）
        cfg["fusion_conf_bg_coeff"] = 0.85

        # --- (C) Normalize peaks using fewer top pixels (reduce saturation) ---
        # robust_v5/v7 默认 top_ratio=0.005（top0.5%），容易把 t_max 拉低导致“饱和”。
        # 这里回调到更保守的 0.001（top0.1%）。
        cfg["nbs_t_norm_top_ratio"] = 0.001
        cfg["wave_sim_max_top_ratio"] = 0.001

        # keep debug
        cfg["save_debug_best"] = True
        return

    elif p == "robust_v10":
        # ============================================================
        # robust_v10: AUC2-protected 的背景抑制版本
        #
        # 动机（对应你当前的现象）：
        #   - 直接把 bg_gate_thr_q 压得太低，确实能把 AUC3 压下来，
        #     但往往会把真实目标也一起压小，导致 AUC2 大幅下降（“顾此失彼”）。
        #   - 因此这里把“主要抑制手段”转移到 **空间支持度抑制**：
        #       *对孤立高分假警更敏感*
        #       *对连通/成团目标更友好*
        #
        # 继承 robust_v8 的核心谱侧与 EBPC-Head 逻辑，只动 test-time 的空间 refine。
        # ============================================================
        apply_infer_preset(cfg, "robust_v8")
        cfg["infer_preset"] = "robust_v10"

        # ----------------------------
        # spatial refine (suppress-only)
        # ----------------------------
        cfg["spatial_refine_enable"] = True
        cfg["spatial_refine_mode"] = "support_suppress"
        cfg["spatial_refine_ksize"] = 9
        cfg["spatial_refine_iters"] = 3
        cfg["spatial_refine_pad"] = "reflect"

        # 用“邻域均值 nb 的分位数”自适应确定阈值（更稳健，不依赖绝对分数尺度）
        cfg["spatial_refine_thr"] = "auto_nb_q"
        cfg["spatial_refine_thr_q"] = 0.95

        cfg["spatial_refine_k"] = 16.0
        cfg["spatial_refine_p"] = 1.0
        cfg["spatial_refine_min_sup"] = 0.05

        # 不额外压低 bg_gate_thr_q（避免误杀目标导致 AUC2 崩）
        # bg_gate_thr_q / fusion_conf_bg_coeff 继续沿用 robust_v8 的默认值即可。

        cfg["save_debug_best"] = True
        return
    elif p == "robust_v11":
        # robust_v11: 更偏向“目标突出 + 背景抑制稳定”的推理预设
        # - 保留 MT-CAMF 作为强背景抑制项（对结构性背景假警更稳）
        # - 关闭 margin_gate（避免不同度量尺度/混合像元导致误抑制目标）
        # - 弱化 bg_gate 的惩罚权重（避免把“目标-背景混合像元”当成背景）
        # - 空间支持抑制改为更保目标的配置（核更小、阈值略低）
        apply_infer_preset(cfg, "robust_v8")

        # 1) 关闭 margin gate（本项目中最容易误伤目标的一项）
        cfg["nbs_use_margin_gate"] = False
        cfg["score_w_margin_gate"] = 0.0

        # 2) 弱化背景 gate（但阈值更严格，仅在“非常像背景”时才抑制）
        cfg["score_w_bg"] = 0.35
        cfg["bg_gate_thr_q"] = 0.95
        cfg["fusion_conf_use_bg"] = False

        # 3) 让 nbs_t 不那么“尖锐”，提升混合像元/边缘像元的目标响应
        cfg["nbs_delta_t"] = 0.15
        cfg["fusion_conf_thr"] = 0.60

        # 4) 空间抑制：更保目标（建议从这里开始扫参）
        cfg["spatial_refine_mode"] = "support_suppress"
        cfg["spatial_refine_ksize"] = 7
        cfg["spatial_refine_iters"] = 3
        cfg["spatial_refine_thr"] = "auto_nb_q"
        cfg["spatial_refine_thr_q"] = 0.92
        cfg["spatial_refine_k"] = 16.0
        cfg["spatial_refine_min_sup"] = 0.05

        # 5) 可选：轻量角度门控（通常有利于背景假警↓）
        cfg["use_angle_gate"] = True
        cfg["angle_gate_w"] = 0.25

    elif p == "robust_v12":
        # ============================================================
        # robust_v12: 进一步“抬高目标(主攻 AUC2↑)”且尽量不抬高背景(AUC3 不爆)
        # ============================================================
        # 背景：
        #   你当前出现了两类 checkpoint：
        #     - AUC1 很高但 AUC2 偏低、AUC3 极低（阈值很稳但目标不够亮）
        #     - AUC2 提升明显但 AUC3 略回升（目标更亮但假警略增加）
        # robust_v12 的做法是 **只在推理后处理阶段做严格单调标定**：
        #   1) sigmoid 的拐点 mu 用“背景感知(bg_q)”策略估计（更贴近背景尾部）；
        #   2) 在 score 的高分尾部再做一次 highlift（只拉高高分，不整体抬背景）。
        # 这些操作都不改变排序，因此 AUC1 保持不变，但更容易把目标推近 1 -> AUC2↑。
        #
        # 适用场景：
        #   - 你更偏向 ep=119 这种 AUC1 高 / AUC3 很低，但想把 AUC2 抬起来；
        #   - 或者想在保持 AUC1 的基础上进一步“让目标更突出”。
        apply_infer_preset(cfg, "robust_v11")
        cfg["infer_preset"] = "robust_v12"

        # --- 1) 背景感知 sigmoid：mu 来自背景子集的高分位数 ---
        cfg["score_post_sigmoid"] = True
        cfg["score_post_sigmoid_mu"] = "bg_q"
        # 用 sim_bg 选背景：取 sim_bg 的 60% 分位作为阈值，≥阈值的像素更像背景
        cfg["score_post_sigmoid_bg_sel_q"] = 0.60
        # 在背景子集上取 score 的 99.5% 分位做 mu（近似背景极端尾部）
        cfg["score_post_sigmoid_bg_mu_q"] = 0.995
        # 兜底：至少保留 0.5% 像素在 mu 之上，避免把目标整体压扁（AUC2 掉）
        cfg["score_post_sigmoid_min_pos_ratio"] = 0.005
        # k 稍大一点，更像“软阈值化”（建议 18~28 扫参）
        cfg["score_post_sigmoid_k"] = 24.0
        cfg["score_post_sigmoid_rescale"] = True

        # --- 2) 高分尾部 highlift：只拉高 score>thr 的部分 ---
        cfg["score_post_highlift"] = True
        cfg["score_post_highlift_thr"] = 0.50
        cfg["score_post_highlift_gamma"] = 2.0

        # --- 3) 建议把 conf 融合 bg_gate 打开：防止“偶然高 sim_t 的背景”被误当强目标 ---
        cfg["fusion_conf_use_bg"] = True
        cfg["fusion_conf_bg_coeff"] = 0.5

        cfg["save_debug_best"] = True
        return

    elif p == "robust_v13":
        # ============================================================
        # robust_v13: robust_v12 + 自动 pow（背景均值锚定，免 sweep）
        # ============================================================
        # 目标：解决你反馈的“不同数据集需要反复扫 pow 才能压住 AUC3 / 拉起 AUC5”的问题。
        #
        # 做法：
        #   - 沿用 robust_v12 的 bg-aware sigmoid + highlift（主要抬 AUC2，尽量不抬 AUC3）
        #   - 额外开启 score_post_pow_auto：
        #       用 sim_bg / gate 选一批“更像背景”的像素作为校准集，
        #       自动二分搜索 pow，使该子集的均值接近 score_post_auto_bg_mean。
        #
        # 说明：
        #   - pow 映射是严格单调的 => 不改变排序 => AUC1 不变
        #   - 不用 GT，也不用为每个数据集手动扫参
        apply_infer_preset(cfg, "robust_v12")
        cfg["infer_preset"] = "robust_v13"

        # ✅ 自动 pow（核心）
        cfg["score_post_pow_auto"] = True

        # auto-pow 的默认参数（一般不需要改；如果你更追求 AUC5 可把 bg_mean 再压小一点）
        cfg["score_post_auto_bg_mean"] = (
            0.0020  # 希望背景均值压到的水平（越小抑制越强）
        )
        cfg["score_post_auto_pow_max"] = 8.0  # pow 搜索上限（防止过激）
        cfg["score_post_auto_pos_ratio"] = (
            0.005  # 排除 top 0.5% 高分像素，避免把目标混到背景校准集
        )
        cfg["score_post_auto_bg_sel_q"] = 0.60  # sim_bg 用于选背景的分位数
        cfg["score_post_auto_gate_low_q"] = (
            0.70  # gate 用于选背景的分位数（越大越宽松）
        )
        cfg["score_post_auto_min_keep"] = 2048  # 背景校准集最少像素数（不足会自动放宽）

        cfg["save_debug_best"] = True
        return
    elif p == "robust":
        # robust: 不做额外修改（沿用 cfg 默认）
        return

    else:
        raise ValueError(f"Unknown infer_preset: {preset}")

    return


def apply_disable_extras(cfg: Dict):
    """一键关闭改良版额外项，用于做严格对齐/消融。"""
    # 训练侧额外损失
    cfg["w_geom"] = 0.0
    cfg["w_proto"] = 0.0
    cfg["w_proto_ortho"] = 0.0
    cfg["w_repulse"] = 0.0
    cfg["w_local_neg"] = 0.0

    # 推理侧额外抑制/融合
    cfg["score_mode"] = "htdmamba"
    cfg["use_target_aug"] = False
    cfg["target_aug_num"] = 0
    cfg["use_bg_osp"] = False
    cfg["use_wavelet_match"] = False
    cfg["use_angle_gate"] = False
    cfg["adaptive_fusion"] = False
    cfg["score_post_norm"] = "none"




# ============================================================
# Ablation helpers（abu-airport-2 消融实验用）
# ============================================================
def _norm_ablation_name(x: Any) -> str:
    """把用户输入/配置里的 ablation 字符串统一成可匹配的形式。

    你这次的消融配置里会用到类似 `FULL(DSM)` 的写法；但如果只做了
    `upper()/replace(' ', '')`，括号会导致匹配失败（ValueError: Unknown ablation_det）。

    因此这里做更鲁棒的清洗：
      - 去空格
      - `-` `/` -> `_`
      - 去掉括号/中括号/花括号：`()[]{}`
    """
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = s.replace(" ", "")
    s = s.replace("-", "_")
    s = s.replace("/", "_")
    for ch in "()[]{}":
        s = s.replace(ch, "")
    return s


def apply_ablation_det(cfg: Dict[str, Any], mode: Any):
    """推理侧模块消融：DSS_ONLY / DSS+CSS / DSS+ACS / FULL(DSM)

    约定（与你论文里的命名对齐）：
      - DSS：目标相似度 +（可选）梯度/小波相似度（都算“谱证据”）
      - CSS：Confuser Suppression（背景原型 gate / margin gate / bg_osp 等）
      - ACS：Angle Consistency（raw 谱角门控 + angle-aware pooling + t0-angle 后抑制）
      - FULL(DSM)：DSS + CSS + ACS 全开
    """
    name = _norm_ablation_name(mode)
    if not name:
        return

    # 兼容多种写法
    if name in ["DSS_ONLY", "DSSONLY", "DSS"]:
        det = "DSS_ONLY"
    elif name in ["DSS+CSS", "DSS_CSS", "DSSCSS"]:
        det = "DSS_CSS"
    elif name in ["DSS+ACS", "DSS_ACS", "DSSACS"]:
        det = "DSS_ACS"
    elif name in ["FULL", "FULL_DSM", "FULLDSM", "DSM", "DSS+CSS+ACS", "DSS_CSS_ACS"]:
        det = "FULL"
    else:
        raise ValueError(f"Unknown ablation_det: {mode}")

    # 统一要求 score_mode 走 DSM 的 nbs_proto（否则 DSS/CSS/ACS 的语义会飘）
    cfg["score_mode"] = "nbs_proto"

    # -----------------------------
    # 先把“可选项”设为一个可控默认，再按 det 覆盖
    # -----------------------------
    # CSS：背景抑制
    cfg.setdefault("use_bg_osp", True)
    cfg.setdefault("score_w_bg", 1.0)
    cfg.setdefault("nbs_use_margin_gate", True)
    cfg.setdefault("score_w_margin_gate", 1.0)
    cfg.setdefault("fusion_conf_use_bg", True)
    cfg.setdefault("fusion_conf_use_margin", False)

    # ACS：谱角一致性
    cfg.setdefault("use_angle_gate", True)
    cfg.setdefault("angle_aware_pool", True)
    cfg.setdefault("fusion_conf_use_angle", True)
    cfg.setdefault("adaptive_angle_gate", True)
    cfg.setdefault("score_post_t0_angle_pow", "auto")
    cfg.setdefault("nbs_t_norm_mask", "angle_t0_gate")
    cfg.setdefault("wave_sim_max_mask", "angle_t0_gate")
    cfg.setdefault("score_post_norm_mask", "angle_t0_gate")

    # -----------------------------
    # 根据 det 模式覆盖
    # -----------------------------
    if det == "DSS_ONLY":
        # 只保留 DSS（谱证据），不做 confuser suppression，也不做角度一致性门控

        # CSS off
        cfg["use_bg_osp"] = False
        cfg["score_w_bg"] = 0.0
        cfg["nbs_use_margin_gate"] = False
        cfg["score_w_margin_gate"] = 0.0
        cfg["fusion_conf_use_bg"] = False
        cfg["fusion_conf_use_margin"] = False

        # ACS off
        cfg["use_angle_gate"] = False
        cfg["angle_aware_pool"] = False
        cfg["fusion_conf_use_angle"] = False
        cfg["adaptive_angle_gate"] = False
        cfg["score_post_t0_angle_pow"] = 0.0

        # 禁用所有 angle-based mask（避免“关 ACS 但仍在用 angle 做标定”的隐性泄露）
        cfg["nbs_t_norm_mask"] = "none"
        cfg["wave_sim_max_mask"] = "none"
        cfg["score_post_norm_mask"] = "none"

    elif det == "DSS_CSS":
        # DSS + CSS：打开 confuser suppression，但不使用角度一致性
        # CSS on：保留 preset/配置里的设置即可

        # ACS off
        cfg["use_angle_gate"] = False
        cfg["angle_aware_pool"] = False
        cfg["fusion_conf_use_angle"] = False
        cfg["adaptive_angle_gate"] = False
        cfg["score_post_t0_angle_pow"] = 0.0
        cfg["nbs_t_norm_mask"] = "none"
        cfg["wave_sim_max_mask"] = "none"
        cfg["score_post_norm_mask"] = "none"

    elif det == "DSS_ACS":
        # DSS + ACS：只用角度一致性，不做背景 confuser suppression

        # CSS off
        cfg["use_bg_osp"] = False
        cfg["score_w_bg"] = 0.0
        cfg["nbs_use_margin_gate"] = False
        cfg["score_w_margin_gate"] = 0.0
        cfg["fusion_conf_use_bg"] = False
        cfg["fusion_conf_use_margin"] = False

        # ACS on（默认已经 setdefault 了，这里仅做显式保证）
        cfg["use_angle_gate"] = True
        cfg["angle_aware_pool"] = True
        cfg["fusion_conf_use_angle"] = True
        cfg["adaptive_angle_gate"] = True
        cfg["score_post_t0_angle_pow"] = cfg.get("score_post_t0_angle_pow", "auto")
        cfg["nbs_t_norm_mask"] = str(cfg.get("nbs_t_norm_mask", "angle_t0_gate"))
        cfg["wave_sim_max_mask"] = str(cfg.get("wave_sim_max_mask", "angle_t0_gate"))
        cfg["score_post_norm_mask"] = str(cfg.get("score_post_norm_mask", "angle_t0_gate"))

    elif det == "FULL":
        # FULL(DSM)：全开（默认 setdefault 的就是全开）
        cfg["use_bg_osp"] = bool(cfg.get("use_bg_osp", True))
        cfg["score_w_bg"] = float(cfg.get("score_w_bg", 1.0))
        cfg["nbs_use_margin_gate"] = bool(cfg.get("nbs_use_margin_gate", True))
        cfg["score_w_margin_gate"] = float(cfg.get("score_w_margin_gate", 1.0))

        cfg["use_angle_gate"] = bool(cfg.get("use_angle_gate", True))
        cfg["angle_aware_pool"] = bool(cfg.get("angle_aware_pool", True))
        cfg["fusion_conf_use_angle"] = bool(cfg.get("fusion_conf_use_angle", True))
        cfg["adaptive_angle_gate"] = bool(cfg.get("adaptive_angle_gate", True))
        cfg["score_post_t0_angle_pow"] = cfg.get("score_post_t0_angle_pow", "auto")

    # 归一化写回，便于日志/结果文件里标识
    cfg["ablation_det"] = det


def apply_ablation_spec(cfg: Dict[str, Any], mode: Any):
    """谱表征/证据路径消融：RAW / RAW+GRAD / RAW+WAVE / RAW+GRAD+WAVE

    注意：这里同时控制两件事：
      1) encoder 侧是否使用 deriv/wavelet token（use_deriv / use_wavelet）
      2) 推理侧是否启用 deriv_match / wavelet_match 作为额外谱证据（use_deriv_match / use_wavelet_match）

    如果你只想 ablate “推理证据”而不动网络结构，可以只改 use_*_match，
    并不要在这里设置 ablation_spec。
    """
    name = _norm_ablation_name(mode)
    if not name:
        return

    if name in ["RAW", "ORI", "ORIG", "ORIGINAL", "ONLY_RAW"]:
        spec = "RAW"
    elif name in ["RAW+GRAD", "RAW+DERIV", "RAW_DERIV", "RAW_GRAD", "RAWGRAD", "RAWD"]:
        spec = "RAW_DERIV"
    elif name in ["RAW+WAVE", "RAW+WAVELET", "RAW_WAVE", "RAW_WAVELET", "RAWWAVE"]:
        spec = "RAW_WAVE"
    elif name in [
        "FULL",
        "RAW+GRAD+WAVE",
        "RAW+DERIV+WAVE",
        "RAW_DERIV_WAVE",
        "RAW_DERIV_WAVELET",
        "ALL",
    ]:
        spec = "FULL"
    else:
        raise ValueError(f"Unknown ablation_spec: {mode}")

    use_deriv = spec in ["RAW_DERIV", "FULL"]
    use_wave = spec in ["RAW_WAVE", "FULL"]

    # -------- encoder 侧开关（需要重新训练才有意义）--------
    cfg["use_deriv"] = bool(use_deriv)
    cfg["use_wavelet"] = bool(use_wave)

    # -------- 推理侧证据开关（同一 ckpt 也可做推理消融）--------
    cfg["use_deriv_match"] = bool(use_deriv)
    cfg["use_wavelet_match"] = bool(use_wave)

    # -------- DSM(nbs_proto) 融合权重 --------
    if not use_wave:
        cfg["score_w_wave"] = 0.0
    else:
        cfg.setdefault("score_w_wave", 0.6)

    # (新增) derivative evidence weight
    if not use_deriv:
        cfg["score_w_deriv"] = 0.0
    else:
        cfg.setdefault("score_w_deriv", 0.35)

    # derivative NBS 参数（如果你不想手动写，这里给默认值）
    cfg.setdefault("nbs_delta_d", 0.12)
    cfg.setdefault("deriv_sim_max", "topk_median")
    cfg.setdefault("deriv_sim_max_top_ratio", 0.001)
    cfg.setdefault("deriv_sim_max_min_k", 16)
    # 默认复用 wave_sim_max 的 mask（通常是 angle_t0_gate）
    cfg.setdefault("deriv_sim_max_mask", cfg.get("wave_sim_max_mask", "angle_t0_gate"))

    cfg["ablation_spec"] = spec


def apply_ablation(cfg: Dict[str, Any]):
    """统一入口：从 cfg 读取 ablation_det / ablation_spec 并应用。

    设计原则：
      - 只需要在配置文件里写清楚“消融名称”，不必手动改一堆 key；
      - 覆盖发生在 preset + 外部 config + CLI 覆盖之后（优先级最高）。
    """
    det = cfg.get("ablation_det", None)
    spec = cfg.get("ablation_spec", None)
    if det is not None:
        apply_ablation_det(cfg, det)
    if spec is not None:
        apply_ablation_spec(cfg, spec)

def build_cfg(args) -> dict:
    # =========================
    # 0) dataset meta
    # =========================
    mat = sio.loadmat(args.path)
    band = int(mat["data"].shape[-1])

    # =========================
    # 1) config
    # =========================
    cfg = {
        # ---------- I/O ----------
        "path": args.path,
        "save_dir": args.save_dir,
        "log_dir": args.log_dir,
        "out_dir": args.out_dir,
        # ---------- runtime ----------
        "device": args.device,
        "seed": args.seed,
        "band": band,
        # ---------- training ----------
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": 1.0,
        # optimizer/scheduler
        # 论文与对比学习常用：warmup + cosine decay；能提升收敛稳定性与复现性
        "use_lr_scheduler": True,
        "lr_warmup_epochs": 10,
        "min_lr_ratio": 0.05,
        # ---------- backbone (HTD-Mamba style) ----------
        "m": 5,  # spectral group length
        "token_stride": None,  # 默认采用 stride = m//4（与 HTD-Mamba 官方实现一致，重叠分组更利于细粒度谱形）
        "token_activation": "leaky_relu",
        "channel": 16,
        "state_size": 16,
        "layer": 1,
        "bidirectional": True,
        # ---------- wavelet / derivatives ----------
        "use_wavelet": True,
        "wavelet_type": "db4",
        "wavelet_level": 2,  # multi-scale SWT: levels 1..2
        "wavelet_learnable": False,
        "use_deriv": True,
        "deriv_scale": 1.0,
        # NEW: smooth-before-derivative (Savitzky-Golay / low-pass)
        "deriv_smooth": "savgol",  # "none" | "gaussian" | "savgol"
        "deriv_smooth_window": 7,
        "deriv_smooth_poly": 2,
        "deriv_gaussian_sigma": 1.0,
        "use_channel_gate": True,
        # 输入光谱归一化：去均值 + L2（与 correlation-cosine 一致，提升亮度/偏置鲁棒性）
        "use_input_norm": True,
        # ---------- model-side spectral drift aug ----------
        # 经验：LosAngeles/ABU 这类窄谱目标检测里，过强 drift 会把“细粒度谱形差异”学成不变性，
        # 反而降低目标可检测性（AUC2）。因此这里默认更保守：
        "drift_p": 0.15,
        # drift 的形变幅度建议不要太大，否则会破坏窄谱保真（导致目标难匹配）
        "drift_shift_range": (-0.75, 0.75),
        "drift_scale_range": (-0.005, 0.005),
        "drift_piecewise_p": 0.35,
        "drift_sinusoid_p": 0.25,
        "drift_seg_mag": 0.008,
        "drift_curriculum": True,
        "drift_warmup": 40,
        # ---------- embedding ----------
        # 对齐 HTD-Mamba 论文实现：embedding size 16, feature vector dim 32
        "embed_dim": 32,
        # ---------- BPSA: background prototypes ----------
        "proto_bank_size": 64,
        "w_proto": 0.15,
        "w_proto_ortho": 0.02,
        # prototype warmup：避免早期原型损失把 embedding 拉塌（只改变权重，不增加训练步骤）
        "proto_start_epoch": 10,
        "proto_warmup": 30,
        "proto_tau": 0.1,
        "sinkhorn_eps": 0.05,
        "sinkhorn_iters": 3,
        # density gate (谱角单位: radian)
        # dens_angle 越小越像“常见背景”
        "proto_density_topk": 8,
        "proto_density_thr": 0.30,  # 约 17°，比你旧版 0.22 更“包容”异质背景
        "proto_density_k": 12.0,
        # density 权重计算方式：
        # - sigmoid : σ(k*(thr - dens_angle))（绝对阈值，角度有物理意义，但跨数据集可能需调）
        # - quantile: 选 dens_angle 最小的一部分做“高置信背景”（更稳健，减少稀有目标污染原型）
        "proto_density_mode": "q_sigmoid",
        "proto_bg_quantile": 0.70,
        # 可选：只用高 w_bg 样本做 Sinkhorn assignment（默认 None=关）
        "proto_bg_only_thr": 0.60,
        "proto_min_bg_samples": 16,
        # ---------- Spectral‑Geometry Alignment (SGA) ----------
        "w_geom": 0.12,
        "geom_topk": 8,
        "geom_sim_thr": 0.70,
        "geom_use_density_weight": False,
        # ---------- outlier repulsion（可选，默认关） ----------
        "w_repulse": 0.0,
        "repulse_start_epoch": 20,
        "repulse_outlier_thr": 0.45,  # dens_angle > 0.45 rad (~25.8°) 视为极端 outlier
        "repulse_outlier_k": 12.0,
        "repulse_margin": 0.55,
        # ---------- data augmentation (window-level, chain_safe) ----------
        "use_augment": True,
        "augment_view1_scale": 0.35,  # weak view
        # strong view 太强会导致“谱形细节被抹平”，默认稍微收敛一点更利于窄谱目标
        "augment_view2_scale": 0.85,  # strong view
        # 增强强度 curriculum（先弱后强）
        "aug_curriculum": True,
        "aug_ramp_epochs": 30,
        "augment_view1_scale_start": 0.20,
        "augment_view2_scale_start": 0.45,
        "augment_ctrl": {
            # 这些 key 会被 Data_Patch._scale_ctrl 按 view1/view2_scale 再缩放
            "calib_p": 0.6,
            "gaussian_p": 0.25,
            "dropout_p": 0.25,
            "mult_p": 0.25,
            "stripe_p": 0.20,
            # NEW: physical-consistent smile / keystone
            "smile_p": 0.15,
            "smile_shift": (-0.8, 0.8),  # band-index units
            "keystone_p": 0.15,
            "keystone_shift_pix": (-0.8, 0.8),  # pixel units (fractional allowed)
            "keystone_axis": "x",  # "x"|"y" (or 1/0)
            "keystone_mode": "linear_sin",
            "keystone_freq_range": (0.5, 1.5),
            # 强度
            "gaussian_sigma": (0.4, 1.2),
            "dropout_w": (3, 12),
            "mult_std": 0.03,
            "stripe_strength": (0.85, 1.15),
            "calib": ((-0.02, 0.02), (-0.03, 0.03), (-0.01, 0.01)),
        },
        # ---------- patch context encode / hard negative ----------
        "w_size": 11,
        "patch_encode_tau": 1.0,  # 重要：SESA 权重建议 tau≈1（太小会退化为中心像素）
        "patch_encode_sigma": 2.0,
        # Top-k 过滤不是 HTD‑Mamba/SASA 的必需项；
        # 如果打开它（如 0.25），会更“保守”地只用最相似邻域做编码。
        # 注意：ctx_angle 的门控我们已经改为基于“完整 patch”计算，因此不会因 top-k 而失效。
        "patch_encode_topk_ratio": None,
        "hard_neg_quantile": 0.10,
        "return_ctx_sim": True,
        # ---------- contrastive temperatures ----------
        # 对齐论文温度：0.1（更稳的对比学习温度，过小会导致梯度更尖锐，复现更敏感）
        "delta": 0.10,
        "delta_ctx": 0.10,
        # ---------- losses ----------
        "w_intra": 1.0,
        "w_ctx": 0.35,
        # ctx 门控（谱角阈值，radian）：角度越小越相似，越信任 ctx loss。
        # 建议用 quantile 模式做“自适应阈值”，减少跨数据集调参。
        "ctx_thr_mode": "quantile",  # "fixed" | "quantile"
        "ctx_thr_q": 0.60,
        "ctx_sim_thr": 0.35,  # fixed 模式才使用
        "ctx_sim_k": 12.0,
        "ctx_start_epoch": 1,
        "ctx_warmup": 20,
        # local hard-negative（三元组）可选：建议先关掉
        "w_local_neg": 0.0,
        "local_margin": 0.2,
        "local_neg_warmup": 10,
        # false-negative suppression
        "fn_topk": 8,
        # =========================
        # 2) Eval / detection config
        # =========================
        "test_load_weight": args.test_load_weight,
        "ts_type": args.ts_type,
        # 评估时相似度 pooling（降低 max over candidates 的偶然高相似假警）
        # 可选："max" | "topk_mean" | "softmax" | "softmax_topk"
        "target_sim_pool": "softmax_topk",  # 比 max 更稳，避免 target-bank 偶然高相似带来的假警
        "target_sim_topk": 3,
        "target_sim_temp": 0.07,
        # angle / wave pooling (optional overrides; default fallback to target_sim_*)
        "angle_sim_pool": "softmax_topk",
        "angle_sim_topk": 3,
        "angle_sim_temp": 0.07,
        "wave_sim_pool": "softmax_topk",
        "wave_sim_topk": 3,
        "wave_sim_temp": 0.07,
        "bg_sim_pool": "softmax_topk",
        "bg_sim_topk": 3,
        "bg_sim_temp": 0.07,
        # ---------- background suppression gate (inference) ----------
        # 原先 nbs_bg=exp(-((sim_bg-1)^2)/delta_bg) 很依赖 sim_bg≈1。
        # 对 SwAV 原型相似度而言通常不满足，因此默认改成 sigmoid 阈值门控更稳。
        # thr="auto"：根据整幅图 sim_bg 的分位数自动确定阈值，提升异质背景下阈值稳定性。
        "bg_gate_mode": "sigmoid_sim",  # "sigmoid_sim" | "nbs"
        "bg_gate_k": 24.0,
        "bg_gate_thr": "auto",
        # 经验：把阈值分位数稍微调低，让更多结构性背景进入抑制区间
        "bg_gate_thr_q": 0.90,
        # htdmamba 基线消融（默认不叠加额外门控）
        "htd_apply_angle_gate": False,
        "htd_apply_wavelet_match": False,
        # nbs_proto 是否额外使用 margin gate（更强背景抑制，默认开）
        "nbs_use_margin_gate": True,
        # 当 target_sim_metric 使用 ACE/MT-CAMF 时，sim_t 与 sim_bg 不在同一尺度；
        # 默认 sem_margin 仍使用余弦(sim_t_cos - sim_bg) 更稳。若你确认需要，可用此开关强制用 override 后的 sim_t 重新计算。
        "sem_margin_use_override_sim": False,
        "score_w_margin_gate": 1.0,
        # eval/select 是否保存可视化输出
        "eval_save_outputs": True,
        "eval_save_debug": True,
        "select_save_outputs": bool(args.select_save_outputs),
        "select_metric": str(args.select_metric).upper(),
        "select_plot_history": True,
        # target-bank augmentation (eval only)
        "target_aug_num": 6,  # eval-only：目标端元 bank 扰动个数（总数=1+num）；提升对端元变异/微漂移的鲁棒性，训练阶段不引入目标谱
        "target_aug_shift_range": (-1.0, 1.0),
        "target_aug_scale_range": (-0.005, 0.005),
        # 正弦扰动幅度建议更保守：过大容易“造出”与背景误匹配的 target
        "target_aug_sine_amp_range": (0.0, 0.06),
        "target_aug_sine_freq_range": (0.5, 1.5),
        # NEW: endmember-bundle style target bank (eval only)
        "target_aug_smooth_sigma_range": (0.0, 0.6),  # bandwidth/resolution variation
        "target_aug_calib": ((-0.01, 0.01), (-0.01, 0.01), (-0.005, 0.005)),
        "target_aug_noise_std": 0.002,
        # target-bank prior weighting (eval only)
        # 让 bank 成员“越偏离 t0 权重越小”，降低多次比较导致的背景假警
        "target_bank_weighting": "angle",  # "none" | "angle"
        "target_bank_weight_sigma": 0.12,
        "target_bank_angle_clip": 0.30,  # rad，超过该偏离直接视作不可信（权重≈0）
        "target_bank_min_weight": 1e-6,
        # background subspace removal (eval only)
        # 用背景原型近似张成子空间，先解释并减去背景成分，再做 target 匹配
        "use_bg_osp": True,
        "bg_osp_k": 4,
        "bg_osp_alpha": 0.8,
        "bg_osp_apply": "both",  # "both" | "x"(only pixel) | "t"(only target)
        # bg_osp 的实现方式：
        # - approx: 近似投影（假设原型近似正交，速度最快）
        # - exact : top-k 原型上做最小二乘投影（更稳健，推荐）
        "bg_osp_method": "exact",
        "bg_osp_ridge": 1e-3,
        # fusion
        "use_wavelet_match": True,
        "wavelet_level_decay": 0.7,  # multi-level SWT fusion: w_l = decay^l
        # score mode
        "score_mode": "nbs_proto",
        # nbs_proto 超参数
        "nbs_delta_t": 0.10,
        "nbs_delta_w": 0.15,
        "nbs_delta_bg": 0.10,
        # wave_sim 的归一化上限：推荐用 topk 统计量（比单点 max / 极高分位数更稳健）
        "wave_sim_max": "topk_median",
        "wave_sim_max_top_ratio": 0.001,
        "wave_sim_max_min_k": 16,
        # target-sim 的 NBS 也支持自标定：topk 统计量更不怕偶然假警 outlier
        "nbs_t_norm": "topk_median",
        "nbs_t_norm_top_ratio": 0.001,
        "nbs_t_norm_min_k": 16,
        # NBS 标定的“高分集”mask：用 raw 谱角门控先剔除明显不合理的背景 outlier，
        # 避免 t_max / wave_max 被拉得过大导致分数压扁（AUC1/AUC2 掉点）。
        # 可选： "none" | "angle_gate" | "angle_raw"
        "nbs_t_norm_mask": "angle_t0_gate",
        "nbs_t_norm_angle_gate_thr": 0.5,
        "wave_sim_max_mask": "angle_t0_gate",
        "wave_sim_max_angle_gate_thr": 0.5,
        # 融合权重（log-space exponents）
        "score_w_t": 1.0,
        "score_w_wave": 0.6,
        "score_w_bg": 1.0,
        # ---------- score post-normalization (inference) ----------
        # AUC1 只看排序，但 AUC2/AUC3 对 τ(阈值) 很敏感。
        # 当多证据融合把整体分数压小后，τ∈[0,1] 上会出现一段“无意义阈值区间”，
        # 导致 AUC2/AUC4/AUC5 下降。
        # 这里提供一个 *单调* 的后处理标定：按 max 或高分位数把 score 拉回 [0,1]。
        # - none: 关闭（更贴近原论文实现）
        # - q   : 用分位数(默认 0.999)做缩放（更稳健，推荐）
        # - max : 用全局最大值缩放（对 outlier 更敏感）
        "score_post_norm": "q",
        "score_post_norm_q": 0.999,
        # 可选：只用“物理合理”的像素子集估计分位数，避免 outlier 拉低标定
        "score_post_norm_mask": "angle_t0_gate",  # none | angle_t0_gate | angle_gate
        "score_post_norm_angle_t0_gate_thr": 0.5,
        "score_post_norm_angle_gate_thr": 0.5,
        "score_post_norm_min_keep": 1024,
        # ---------- raw 谱角门控（推理期） ----------
        "use_angle_gate": True,
        "angle_gate_thr": 0.30,  # rad (~17°)
        "angle_gate_k": 12.0,
        "angle_gate_w": 1.0,
        # (新增) angle-aware pooling：用 raw 谱角对 target-bank 做逐像素门控权重（建议开启）
        "angle_aware_pool": True,
        "angle_pool_thr": 0.30,  # 默认复用 angle_gate_thr
        "angle_pool_k": 12.0,  # 默认复用 angle_gate_k
        "angle_pool_min_w": 1e-6,
        # (新增) adaptive fusion：强目标像素减弱抑制项，避免 gate “误伤”导致 AUC1 掉点
        "adaptive_fusion": True,
        "fusion_conf_thr": 0.65,
        "fusion_conf_k": 12.0,
        "fusion_min_sup": 0.10,  # sup 的下界（越小越接近“强目标只看 nbs_t”）
        # (新增) adaptive fusion 的 conf 不仅看 nbs_t，还融合更“稳”的物理/背景信号，
        # 防止“偶然高 sim_t 的背景 outlier”被误判为强目标，从而让抑制项失效。
        # 这里采用温和融合：factor ∈ [1-coef, 1]，不至于把真目标的 conf 拉得过低。
        "fusion_conf_use_angle": True,
        "fusion_conf_angle_source": "t0",  # "t0"(更严格) | "bank"(更宽松)
        "fusion_conf_ang_coeff": 0.5,
        "fusion_conf_use_bg": True,
        "fusion_conf_bg_coeff": 0.5,
        # 可选：margin 也参与 conf（一般不需要，先关）
        "fusion_conf_use_margin": False,
        "fusion_conf_margin_coeff": 0.25,
        "adaptive_angle_gate": True,
        # margin_zscore 模式（保留）
        "w_sem": 2.0,
        "w_wave": 0.8,
        "gate_k": 8.0,
        # margin gate 的阈值：推理期自适应（不增加训练复杂度）
        "margin_thr": "auto_mad",
        "margin_thr_k_mad": 2.5,
        "margin_thr_clip": (-0.5, 0.5),
        "score_temp": 1.0,
        "score_z_clip": 8.0,
        # debug eval
        "debug_eval_interval": 0,
    }

    # 如果用户想改 group_length，在这里自动同步 token_stride
    cfg["token_stride"] = (
        int(args.token_stride)
        if args.token_stride is not None
        else max(1, int(cfg["m"]) // 4)
    )

    # ---------------- presets & overrides ----------------
    cfg.setdefault("patch_encode_sim", "corr")
    cfg.setdefault("eval_save_overlay", True)

    # 1) 预设（先应用预设，再应用 CLI 显式覆盖）
    apply_train_preset(cfg, getattr(args, "train_preset", "robust"))
    apply_infer_preset(cfg, getattr(args, "infer_preset", "robust_v2"))

    # 2) CLI 显式覆盖（仅当参数不是 None 时才覆盖）
    if getattr(args, "embed_dim", None) is not None:
        cfg["embed_dim"] = int(args.embed_dim)
    if getattr(args, "channel", None) is not None:
        cfg["channel"] = int(args.channel)
    if getattr(args, "state_size", None) is not None:
        cfg["state_size"] = int(args.state_size)
    if getattr(args, "layer", None) is not None:
        cfg["layer"] = int(args.layer)

    if getattr(args, "w_intra", None) is not None:
        cfg["w_intra"] = float(args.w_intra)
    if getattr(args, "w_ctx", None) is not None:
        cfg["w_ctx"] = float(args.w_ctx)
    if getattr(args, "w_geom", None) is not None:
        cfg["w_geom"] = float(args.w_geom)
    if getattr(args, "w_proto", None) is not None:
        cfg["w_proto"] = float(args.w_proto)
    if getattr(args, "proto_start_epoch", None) is not None:
        cfg["proto_start_epoch"] = int(args.proto_start_epoch)
    if getattr(args, "proto_warmup", None) is not None:
        cfg["proto_warmup"] = int(args.proto_warmup)

    if getattr(args, "fn_topk", None) is not None:
        cfg["fn_topk"] = int(args.fn_topk)

    if getattr(args, "patch_encode_sim", None) is not None:
        cfg["patch_encode_sim"] = str(args.patch_encode_sim).lower()
    if getattr(args, "patch_encode_sigma", None) is not None:
        cfg["patch_encode_sigma"] = float(args.patch_encode_sigma)
    if getattr(args, "patch_encode_topk_ratio", None) is not None:
        cfg["patch_encode_topk_ratio"] = float(args.patch_encode_topk_ratio)

    if getattr(args, "drift_p", None) is not None:
        cfg["drift_p"] = float(args.drift_p)
    if getattr(args, "drift_warmup", None) is not None:
        cfg["drift_warmup"] = int(args.drift_warmup)

    if getattr(args, "aug_view1_scale", None) is not None:
        cfg["augment_view1_scale"] = float(args.aug_view1_scale)
    if getattr(args, "aug_view2_scale", None) is not None:
        cfg["augment_view2_scale"] = float(args.aug_view2_scale)
    if getattr(args, "aug_view1_scale_start", None) is not None:
        cfg["augment_view1_scale_start"] = float(args.aug_view1_scale_start)
    if getattr(args, "aug_view2_scale_start", None) is not None:
        cfg["augment_view2_scale_start"] = float(args.aug_view2_scale_start)
    if getattr(args, "aug_ramp_epochs", None) is not None:
        cfg["aug_ramp_epochs"] = int(args.aug_ramp_epochs)

    if getattr(args, "use_wavelet", None) is not None:
        cfg["use_wavelet_match"] = bool(args.use_wavelet)
    if getattr(args, "use_deriv", None) is not None:
        cfg["use_deriv_match"] = bool(args.use_deriv)

    # --- 推理端关键超参（可单独 sweep，不需要重训） ---
    if getattr(args, "nbs_t_norm_top_ratio", None) is not None:
        cfg["nbs_t_norm_top_ratio"] = float(args.nbs_t_norm_top_ratio)
    if getattr(args, "nbs_delta_t", None) is not None:
        cfg["nbs_delta_t"] = float(args.nbs_delta_t)
    if getattr(args, "fusion_conf_thr", None) is not None:
        cfg["fusion_conf_thr"] = float(args.fusion_conf_thr)

    if getattr(args, "bg_osp_alpha", None) is not None:
        cfg["bg_osp_alpha"] = float(args.bg_osp_alpha)
    if getattr(args, "bg_gate_thr_q", None) is not None:
        cfg["bg_gate_thr_q"] = float(args.bg_gate_thr_q)
    if getattr(args, "bg_gate_k", None) is not None:
        cfg["bg_gate_k"] = float(args.bg_gate_k)
    if getattr(args, "gate_k", None) is not None:
        cfg["gate_k"] = float(args.gate_k)

    if getattr(args, "score_w_bg", None) is not None:
        cfg["score_w_bg"] = float(args.score_w_bg)
    if getattr(args, "score_w_wave", None) is not None:
        cfg["score_w_wave"] = float(args.score_w_wave)
    if getattr(args, "score_w_margin", None) is not None:
        cfg["score_w_margin_gate"] = float(args.score_w_margin)

    if getattr(args, "score_post_norm", None) is not None:
        cfg["score_post_norm"] = str(args.score_post_norm).lower()
    if getattr(args, "score_post_norm_q", None) is not None:
        cfg["score_post_norm_q"] = float(args.score_post_norm_q)
    if getattr(args, "score_post_norm_mask", None) is not None:
        cfg["score_post_norm_mask"] = str(args.score_post_norm_mask).lower()
    if getattr(args, "score_post_norm_min_keep", None) is not None:
        cfg["score_post_norm_min_keep"] = int(args.score_post_norm_min_keep)

    if getattr(args, "score_post_pow", None) is not None:
        cfg["score_post_pow"] = float(args.score_post_pow)

    if getattr(args, "score_post_pow_auto", None) is not None:
        cfg["score_post_pow_auto"] = bool(args.score_post_pow_auto)
    if getattr(args, "score_post_auto_bg_mean", None) is not None:
        cfg["score_post_auto_bg_mean"] = float(args.score_post_auto_bg_mean)
    if getattr(args, "score_post_auto_pow_max", None) is not None:
        cfg["score_post_auto_pow_max"] = float(args.score_post_auto_pow_max)
    if getattr(args, "score_post_auto_pos_ratio", None) is not None:
        cfg["score_post_auto_pos_ratio"] = float(args.score_post_auto_pos_ratio)

    # conformal calibration (optional)
    if getattr(args, "conformal_gate_thr", None) is not None:
        cfg["conformal_gate_thr"] = float(args.conformal_gate_thr)
    if getattr(args, "conformal_gate_q", None) is not None:
        cfg["conformal_gate_q"] = float(args.conformal_gate_q)
    if getattr(args, "conformal_min_keep", None) is not None:
        cfg["conformal_min_keep"] = int(args.conformal_min_keep)

    if getattr(args, "angle_gate_k", None) is not None:
        cfg["angle_gate_k"] = float(args.angle_gate_k)
    if getattr(args, "angle_gate_w", None) is not None:
        cfg["angle_gate_w"] = float(args.angle_gate_w)

    if getattr(args, "eval_save_outputs", None) is not None:
        cfg["eval_save_outputs"] = bool(args.eval_save_outputs)
    if getattr(args, "eval_save_debug", None) is not None:
        cfg["eval_save_debug"] = bool(args.eval_save_debug)
    if getattr(args, "eval_save_overlay", None) is not None:
        cfg["eval_save_overlay"] = bool(args.eval_save_overlay)

    # 3) 一键关 extras（放在最后，确保它覆盖上面所有设置）
    if getattr(args, "disable_extras", False):
        apply_disable_extras(cfg)

    # 保存一下，方便日志里追溯
    cfg["train_preset"] = str(getattr(args, "train_preset", "robust"))
    cfg["infer_preset"] = str(getattr(args, "infer_preset", "robust_v2"))

    return cfg


def parse_args():
    p = argparse.ArgumentParser(
        description="HTD-Mamba Improved (Los Angeles / abu-airport-2)"
    )

    # ---------------- basic ----------------
    p.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "eval", "select", "sweep", "all"],
        help="train:只训练；select:挑最优；eval:评估；sweep:一次评估批量扫参；all:训练+挑最优+评估",
    )
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="外部配置文件路径(.py/.json)，用于覆盖脚本默认 cfg（更利于复现/对比实验）",
    )

    p.add_argument(
        "--path",
        type=str,
        default="datasets/abu-airport-2.mat",
        help="数据集 .mat 路径（abu-airport-2 是 Los Angeles 的别名）",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=1)

    # ---------------- training schedule ----------------
    p.add_argument("--epoch", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--eval_batch_size", type=int, default=2048)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # ---------------- dirs ----------------
    p.add_argument("--save_dir", type=str, default="./models")
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument("--out_dir", type=str, default="./result")

    # ---------------- misc ----------------
    p.add_argument(
        "--token_stride",
        type=int,
        default=None,
        help="谱分组 token stride；None 表示用默认 group_length//4（更细粒度）",
    )
    p.add_argument(
        "--test_load_weight",
        type=str,
        default=None,
        help="评估/可视化时指定 ckpt 文件名（例如 ckpt_200_.pt），None 表示用 select_best 的最优",
    )
    p.add_argument(
        "--ts_type",
        type=int,
        default=7,
        help="目标谱选取方式（见 augment.ts_generation）；7=最接近均值谱的目标像素",
    )
    p.add_argument(
        "--select_metric",
        type=str,
        default="AUC4",
        help="select_best 用哪个指标挑最优（AUC1/AUC4 等）",
    )
    p.add_argument(
        "--select_save_outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="挑 ckpt 时是否保存中间可视化（会慢很多）",
    )

    # ================= 新增：预设 =================
    p.add_argument(
        "--train_preset",
        type=str,
        default="robust",
        choices=["robust", "paper", "converge"],
        help="训练侧预设：robust=改良版默认；paper=更贴近论文 HTD-Mamba（SESA 主导，用于对齐 0.99+ AUC1）",
    )
    p.add_argument(
        "--infer_preset",
        type=str,
        default="robust_v2",
        choices=[
            "robust_v13",
            "robust_v12",
            "robust_v11",
            "robust_v10",
            "robust_v9",
            "robust_v8",
            "robust_v7",
            "robust_v6",
            "robust_v5",
            "robust_v4",
            "robust_v3",
            "robust_v2",
            "robust",
            "balanced",
            "balanced_v2",
            "auc1",
            "paper",
        ],
        help="推理侧预设：robust_v12=robust_v11 + 背景感知sigmoid(mu=bg_q)+highlift(只抬高高分尾部，主攻AUC2↑且AUC1不变)；robust_v11=robust_v8 + 关闭 margin gate + 弱化 bg gate + 更保目标的 support_suppress（主攻 目标突出/背景抑制平衡）；robust_v10=robust_v8 + 自适应支持度抑制(仅抑制不扩散，主攻 AUC3↓ 且尽量不伤 AUC2)；robust_v9=robust_v8 + support_suppress + 更强背景 gate（更偏 AUC3↓ 的激进版）；robust_v8=robust_v7 + MT-CAMF + 更保守 bank pooling（主攻结构性背景假警/AUC3↓）；robust_v7=robust_v6 + AdaptiveSigmoidCalibration(单调S形标定，主攻 AUC2↑/AUC3↓ 且 AUC1 不变)；robust_v6=robust_v5 + SpatialConsensusRefine(测试期空间一致性细化，填洞+抑制孤立峰)；robust_v5=Consensus-BOP NBS(像素侧部分 OSP + endmember-bundle/mix-bank + wavelet/deriv 证据，推理期-only，主攻 AUC1 回升且保持低假警)；robust_v4=MT-CAMF(聚类背景统计+Mixture-Tuned+mix-bank，推理期-only，主攻 AUC1 尾部 FN + 异质背景稳定)；robust_v3=ACE 核心证据(更稳 AUC1)+单调幂次压背景；robust_v2=多证据融合(更偏 AUC2+阈值稳定)；auc1/paper=贴近论文基线；balanced/robust=传统抑制/融合",
    )

    # ================= 新增：可覆盖的关键超参（不写则走 cfg 默认/预设） =================
    # model capacity
    p.add_argument("--embed_dim", type=int, default=None)
    p.add_argument("--channel", type=int, default=None)
    p.add_argument("--state_size", type=int, default=None)
    p.add_argument("--layer", type=int, default=None)

    # loss weights
    p.add_argument("--w_intra", type=float, default=None)
    p.add_argument("--w_ctx", type=float, default=None)
    p.add_argument("--w_geom", type=float, default=None)
    p.add_argument("--w_proto", type=float, default=None)
    p.add_argument("--proto_start_epoch", type=int, default=None)
    p.add_argument("--proto_warmup", type=int, default=None)

    # SESA / context encoding
    p.add_argument(
        "--fn_topk",
        type=int,
        default=None,
        help="false-negative mask 的 top-k（0 表示不屏蔽任何负样本）",
    )
    p.add_argument(
        "--patch_encode_sim",
        type=str,
        default=None,
        choices=["cosine", "corr", "cos", "raw"],
        help="SESA patch 编码时的相似度：cosine=论文/原版；corr=减均值余弦（对亮度更稳健）",
    )
    p.add_argument(
        "--patch_encode_sigma",
        type=float,
        default=None,
        help="SESA 空间高斯权重 sigma；None=不加空间先验（更接近原版）",
    )
    p.add_argument(
        "--patch_encode_topk_ratio",
        type=float,
        default=None,
        help="SESA top-k 比例；None=不裁剪（更接近原版）",
    )

    # data-side robustness curriculum
    p.add_argument(
        "--drift_p",
        type=float,
        default=None,
        help="光谱漂移最终概率（curriculum 会从 0 warmup 到 drift_p）",
    )
    p.add_argument("--drift_warmup", type=int, default=None, help="漂移 warmup 轮数")
    p.add_argument(
        "--aug_view1_scale", type=float, default=None, help="弱视图增强强度（最终值）"
    )
    p.add_argument(
        "--aug_view2_scale", type=float, default=None, help="强视图增强强度（最终值）"
    )
    p.add_argument(
        "--aug_view1_scale_start", type=float, default=None, help="弱视图增强起始强度"
    )
    p.add_argument(
        "--aug_view2_scale_start", type=float, default=None, help="强视图增强起始强度"
    )
    p.add_argument(
        "--aug_ramp_epochs", type=int, default=None, help="增强强度 ramp 轮数"
    )

    # feature toggles
    p.add_argument(
        "--use_wavelet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用小波细粒度谱形分支",
    )
    p.add_argument(
        "--use_deriv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用一阶/二阶导数分支",
    )

    # ===== 推理端：可覆盖关键参数（不写则走 preset/default）=====
    p.add_argument(
        "--nbs_t_norm_top_ratio",
        type=float,
        default=None,
        help="NBS-T 归一化 top_ratio（ABU 推荐 0.010~0.020；balanced_v2 默认 0.012）",
    )
    p.add_argument(
        "--nbs_delta_t",
        type=float,
        default=None,
        help="NBS-T 的 delta（越大越不挑剔，AUC2↑但 AUC5 可能↓）",
    )
    p.add_argument(
        "--fusion_conf_thr",
        type=float,
        default=None,
        help="adaptive_fusion 的置信阈值（默认 0.65）",
    )

    p.add_argument(
        "--bg_osp_alpha",
        type=float,
        default=None,
        help="bg-OSP residual 的 alpha（0~1）",
    )
    p.add_argument(
        "--bg_gate_thr_q", type=float, default=None, help="bg_gate 阈值分位数（0~1）"
    )
    p.add_argument(
        "--bg_gate_k", type=float, default=None, help="bg_gate sigmoid 的斜率"
    )
    p.add_argument(
        "--gate_k", type=float, default=None, help="margin_gate sigmoid 的斜率"
    )

    p.add_argument(
        "--score_w_bg", type=float, default=None, help="score 中 bg_gate 的权重"
    )
    p.add_argument(
        "--score_w_wave", type=float, default=None, help="score 中 wavelet NBS 的权重"
    )
    p.add_argument(
        "--score_w_margin", type=float, default=None, help="score 中 margin_gate 的权重"
    )

    # 推理端：score 后处理标定（单调映射，AUC1 不变，但 AUC2/AUC3 更稳定）
    p.add_argument(
        "--score_post_norm",
        type=str,
        default=None,
        choices=["none", "max", "q", "quantile", "conformal"],
        help="score 后处理标定：none/max/q/conformal。q=用高分位数缩放，把分数拉回 [0,1]；conformal=用背景校准样本做经验 p-value 映射，更利于阈值稳定",
    )
    p.add_argument(
        "--score_post_norm_q",
        type=float,
        default=None,
        help="score_post_norm=q 时的分位数（默认 0.999）",
    )
    p.add_argument(
        "--score_post_norm_mask",
        type=str,
        default=None,
        choices=["none", "angle_t0_gate", "angle_gate"],
        help="估计分位数时的像素子集：none/angle_t0_gate/angle_gate",
    )
    p.add_argument(
        "--score_post_norm_min_keep",
        type=int,
        default=None,
        help="mask 子集最少像素数，不足则回退全图（默认 1024）",
    )

    # 单调幂次变换：不改变排序(AUC1)，但通常能把背景的中等分数压向 0，改善 AUC3/AUC5。
    # 经验：abu-airport-2 上 1.3~2.0 往往更稳（默认由 infer_preset 控制）。
    p.add_argument(
        "--score_post_pow",
        type=float,
        default=None,
        help="score 后处理的幂次变换参数（pow>0，默认 1.0=关闭；pow>1 压背景；pow<1 抬分）",
    )
    # 自动 pow（免 sweep）：根据背景子集的分数均值自动选择 pow，用于压低 AUC3 / 提升 AUC5
    p.add_argument(
        "--score_post_pow_auto",
        action="store_true",
        default=None,
        help="启用自动 pow 标定（不需要手动 sweep；依赖 sim_bg/gate 估计背景分布）",
    )
    p.add_argument(
        "--score_post_auto_bg_mean",
        type=float,
        default=None,
        help="自动 pow：希望背景均值压到的水平（默认 0.002）",
    )
    p.add_argument(
        "--score_post_auto_pow_max",
        type=float,
        default=None,
        help="自动 pow：pow 搜索上限（默认 8.0）",
    )
    p.add_argument(
        "--score_post_auto_pos_ratio",
        type=float,
        default=None,
        help="自动 pow：排除 top-K 高分像素的比例（默认 0.005）",
    )

    # score_post_norm=conformal 的可选超参（不填则用 CL_Train 内默认值）
    p.add_argument(
        "--conformal_gate_thr",
        type=float,
        default=None,
        help="conformal 标定：用于选背景校准样本的 gate 阈值（默认 0.5；越小越严格）",
    )
    p.add_argument(
        "--conformal_gate_q",
        type=float,
        default=None,
        help="conformal 标定：当 gate_thr 不足时回退到 gate 的分位数（默认 0.70）",
    )
    p.add_argument(
        "--conformal_min_keep",
        type=int,
        default=None,
        help="conformal 标定：背景校准样本最少保留数量（默认 2048）",
    )

    p.add_argument(
        "--angle_gate_k", type=float, default=None, help="raw-angle gate sigmoid 的斜率"
    )
    p.add_argument(
        "--angle_gate_w",
        type=float,
        default=None,
        help="raw-angle gate 在 score 中的权重",
    )

    # eval/vis
    p.add_argument(
        "--eval_save_outputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="评估阶段是否保存可视化/调试文件",
    )
    p.add_argument(
        "--eval_save_debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="评估阶段是否保存 debug map（sim_bg/margin/gate 等）",
    )
    p.add_argument(
        "--eval_save_overlay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="评估阶段是否额外保存 RGB+score 叠加图（overlay）",
    )

    # ---------------- ablation (optional) ----------------
    p.add_argument(
        "--ablation_det",
        type=str,
        default=None,
        help="推理侧模块消融：DSS_ONLY | DSS+CSS | DSS+ACS | FULL(DSM)。也可在 config 里写 ablation_det",
    )
    p.add_argument(
        "--ablation_spec",
        type=str,
        default=None,
        help="谱表征/证据路径消融：RAW | RAW+GRAD | RAW+WAVE | FULL。也可在 config 里写 ablation_spec",
    )

    # quick ablation switch
    p.add_argument(
        "--disable_extras",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="一键关闭改良版的额外项（proto/geom/bg_osp/angle_gate/wavelet_match/target_aug 等），便于对齐论文基线",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # -------------------- load external config (optional) --------------------
    user_cfg: Dict[str, Any] = {}
    if getattr(args, "config", ""):
        user_cfg = load_user_config(args.config)
        # 将配置里与 argparse 同名的字段写回 args（保证 build_cfg 能吃到这些关键信息）
        arg_keys = {
            "path",
            "save_dir",
            "log_dir",
            "out_dir",
            "device",
            "seed",
            "epoch",
            "batch_size",
            "eval_batch_size",
            "lr",
            "weight_decay",
            "train_preset",
            "infer_preset",
            "ts_type",
            "test_load_weight",
        }
        for k in arg_keys:
            if (k in user_cfg) and hasattr(args, k):
                setattr(args, k, user_cfg[k])

    # create dirs early
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = build_cfg(args)
    # 外部配置优先级最高：在 preset 之后覆盖
    if user_cfg:
        cfg.update(user_cfg)
        cfg["config_path"] = args.config

    # -------------------- apply ablation (optional) --------------------
    # 说明：
    #   - 可以在配置文件里写 ablation_det / ablation_spec；
    #   - 也可以用 CLI 覆盖（CLI 优先级更高）。
    if getattr(args, "ablation_det", None) is not None:
        cfg["ablation_det"] = args.ablation_det
    if getattr(args, "ablation_spec", None) is not None:
        cfg["ablation_spec"] = args.ablation_spec

    # 应用消融映射（放在所有覆盖之后，确保它的优先级最高）
    apply_ablation(cfg)


    if args.mode in ["train", "all"]:
        print("[RUN] Start training...")
        train(cfg)

    if args.mode in ["select", "all"]:
        metric = str(cfg.get("select_metric", "AUC4")).upper()
        print(f"[RUN] Selecting best checkpoint by {metric}...")
        best_score, best_ep = select_best(cfg)
        print(f"Best {metric}={best_score:.6f} at epoch {best_ep}")
        cfg["test_load_weight"] = f"ckpt_{best_ep}_.pt"

    if args.mode in ["eval", "all"]:
        print("[RUN] Evaluating...")
        eval(cfg, save_outputs=bool(cfg.get("eval_save_outputs", True)))

    if args.mode in ["sweep"]:
        print("[RUN] Sweep evaluating (post-calibration grid)...")
        csv_path = sweep_eval(cfg)
        print(f"[RUN] Sweep results saved to: {csv_path}")


if __name__ == "__main__":
    main()
