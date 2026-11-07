# abu-airport-2 消融实验补丁说明

> **v4 修复点（针对你反馈的“命令/配置不匹配导致仍然报错”）：**
>
> - 修复 `ablation_det: FULL(DSM)` 因括号导致无法匹配的问题（现在 `FULL(DSM)` / `FULL_DSM` / `FULL` 都能识别）。
> - `augment.py` 的小波依赖 `pywt` 现在是**可选**：不做小波消融时不会因为缺库直接崩。
> - `CL_Train.py` 增加 **sklearn 缺失 fallback**（用 numpy 计算 ROC/AUC）。
> - `eval` 阶段如果没显式指定 `test_load_weight`，会尝试在 `save_dir` 下自动找最新的 `ckpt_*.pt`。
>


本补丁主要做了两件事：

1. **CL_Train.py**：在 `score_mode == "nbs_proto"`（DSM/NBS 路线）里新增了 **谱梯度(deriv) 相似度证据** 的 NBS 融合项：
   - 新增配置项：`use_deriv_match` / `score_w_deriv` / `nbs_delta_d` / `deriv_sim_max*`
   - 这样就可以做：仅原谱 / 原谱+谱梯度 / 原谱+小波 / 原谱+谱梯度+小波 的消融

2. **CL_Main_LosAngeles.py**：新增配置驱动的消融开关（不需要手动改一堆 key）：
   - `ablation_det`: `DSS_ONLY` / `DSS+CSS` / `DSS+ACS` / `FULL(DSM)`
   - `ablation_spec`: `RAW` / `RAW+GRAD` / `RAW+WAVE` / `FULL`

同时提供了 8 份配置文件（`configs/` 目录），可直接运行。

---

## 目录结构

将本补丁中的文件覆盖到你的工程里：

- 覆盖：
  - `CL_Main_LosAngeles.py`
  - `CL_Train.py`
- 新增：
  - `configs/abu_airport2_abl_det_*.py`
  - `configs/abu_airport2_abl_spec_*.py`

---

## 推荐运行方式（两组消融）

### A. 推理模块消融（DSS / CSS / ACS）

> 推荐：**只训练一次 FULL(DSM)**，然后复用同一个 ckpt 做 `--mode eval` 推理消融。

1) 训练 FULL(DSM) 并自动 select：

```bash
python CL_Main_LosAngeles.py --mode all --config configs/abu_airport2_abl_det_FULL_DSM.py
```

训练完后，在 `./ckpt/abu-airport-2/FULL_DSM/` 下会有若干 `ckpt_*.pt`。

2) 对同一个 ckpt 做推理消融（举例：用你选出来的 best ckpt 文件名替换）：

> ✅ 兼容性说明：`--test_load_weight` 既支持只填文件名（如 `ckpt_160_.pt`），也支持直接填完整/相对路径（如 `./ckpt/abu-airport-2/FULL_DSM/ckpt_160_.pt`）。

```bash
# DSS_ONLY
python CL_Main_LosAngeles.py --mode eval --config configs/abu_airport2_abl_det_DSS_ONLY.py --test_load_weight ./ckpt/abu-airport-2/FULL_DSM/ckpt_xxx.pt

# DSS+CSS
python CL_Main_LosAngeles.py --mode eval --config configs/abu_airport2_abl_det_DSS_CSS.py --test_load_weight ./ckpt/abu-airport-2/FULL_DSM/ckpt_xxx.pt

# DSS+ACS
python CL_Main_LosAngeles.py --mode eval --config configs/abu_airport2_abl_det_DSS_ACS.py --test_load_weight ./ckpt/abu-airport-2/FULL_DSM/ckpt_xxx.pt

# FULL(DSM)（如果想只 eval，不再训练）
python CL_Main_LosAngeles.py --mode eval --config configs/abu_airport2_abl_det_FULL_DSM.py --test_load_weight ./ckpt/abu-airport-2/FULL_DSM/ckpt_xxx.pt
```

> 注意：det 消融配置都把 `save_dir` 指向同一个 `FULL_DSM` 目录，所以 **不要用 `--mode all` 跑 DSS_ONLY/DSS_CSS/DSS_ACS**，否则会把 ckpt 覆盖掉。

---

### B. 谱表征/证据路径消融（RAW / GRAD / WAVE）

这一组需要分别训练（因为 `ablation_spec` 会改变 encoder 结构：`use_deriv/use_wavelet`）。

```bash
# 只有原谱
python CL_Main_LosAngeles.py --mode all --config configs/abu_airport2_abl_spec_RAW.py

# 原谱 + 谱梯度
python CL_Main_LosAngeles.py --mode all --config configs/abu_airport2_abl_spec_RAW_GRAD.py

# 原谱 + 小波
python CL_Main_LosAngeles.py --mode all --config configs/abu_airport2_abl_spec_RAW_WAVE.py

# 原谱 + 谱梯度 + 小波（FULL）
python CL_Main_LosAngeles.py --mode all --config configs/abu_airport2_abl_spec_FULL.py
```

---

## 额外说明

- 如果你只想做“推理证据”消融而不想改网络结构（不需要重训），不要使用 `ablation_spec`，
  而是直接在配置里手动改：`use_deriv_match / use_wavelet_match / score_w_deriv / score_w_wave`。
- 如果你要在 `evidence_z` 路线上做同名消融，也可以扩展 `apply_ablation_det()` 的映射逻辑。