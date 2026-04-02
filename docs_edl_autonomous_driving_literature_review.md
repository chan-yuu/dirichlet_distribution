# EDL 在自动驾驶中的文献综述与路线判定（截至 2026-04-02）

## 1. 你的问题（明确化）
你关心的是：
1. EDL（Evidential Deep Learning）高质量文献有哪些。
2. “用 EDL 改进自动驾驶算法训练”这条路线是否已经被做过。
3. 如果做过，做到哪一层；如果没做透，你的可发表空间在哪里。

本综述给出的是“可用于研究决策”的结论，而不是泛泛论文罗列。

---

## 2. 检索与筛选方法

### 2.1 时间戳
- 检索日期：`2026-04-02`

### 2.2 检索源
- NeurIPS / PMLR / CVPR OpenAccess（优先）
- arXiv（用于跟踪近两年新工作）
- OpenReview（工作坊/在审）
- PubMed（用于核验期刊文章元信息）

### 2.3 关键词
- `evidential deep learning`, `dirichlet uncertainty`, `autonomous driving evidential`
- `trajectory prediction`, `occupancy`, `3D object detection`, `MPC`
- `TransFuser`, `diffusion planning`, `end-to-end autonomous driving uncertainty`

### 2.4 纳入标准
- 优先纳入：顶会/期刊正式发表。
- 预印本只在“直接相关且能说明趋势”时纳入，并明确标注为 preprint。

---

## 3. EDL 基础文献（高质量，必须读）

| 文献 | 级别 | 你该关注的要点 |
|---|---|---|
| Sensoy et al., 2018, *Evidential Deep Learning to Quantify Classification Uncertainty* ([NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html), [arXiv](https://arxiv.org/abs/1806.01768)) | A | 把分类输出建模为 Dirichlet 证据分布，给出“类别概率 + 证据强度”的统一框架。 |
| Amini et al., 2020, *Deep Evidential Regression* ([NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html), [arXiv](https://arxiv.org/abs/1910.02600)) | A | 把 EDL 扩展到回归（Normal-Inverse-Gamma），直接输出 aleatoric + epistemic。 |
| Malinin & Gales, 2018, *Predictive Uncertainty Estimation via Prior Networks* ([NeurIPS](https://papers.neurips.cc/paper_files/paper/2018/hash/3ea2db50e62ceefceaf70f689bf8f9f3-Abstract.html), [arXiv](https://arxiv.org/abs/1802.10501)) | A | Dirichlet 输出在 OOD/分布偏移中的先验建模思路，是 EDL 路线的重要参照系。 |
| 综述：*A Comprehensive Survey on Evidential Deep Learning and Its Applications* ([arXiv 2024](https://arxiv.org/abs/2409.04720)) | B | 了解 EDL 全景与应用谱系，适合快速补齐文献地图。 |
| Kang et al., 2025, *Revisiting the Essential and Nonessential Settings in EDL* ([TPAMI 2025 元信息](https://pubmed.ncbi.nlm.nih.gov/40569804/)) | A | 说明 EDL 的有效性依赖关键设定（损失、正则、训练细节），不是“换个 loss 就稳赢”。 |

结论：EDL 理论和方法本身是成熟的，但工程效果强依赖训练细节与任务设定。

---

## 4. 自动驾驶相关文献（按链路分层）

## 4.1 感知 / 地图 / 占据（最成熟）

| 文献 | 场景 | 与你路线关系 |
|---|---|---|
| *Accurate Training Data for Occupancy Map Prediction in Automated Driving Using Evidence Theory* ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Kalble_Accurate_Training_Data_for_Occupancy_Map_Prediction_in_Automated_Driving_CVPR_2024_paper.pdf)) | 占据图训练数据构建 | 已经把 evidence theory 用到自动驾驶占据图训练。 |
| *EvOcc: Accurate Semantic Occupancy for Automated Driving Using Evidence Theory* ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.pdf)) | 语义占据预测 | 证据化不确定性在占据任务已有强势落地。 |
| *A Simulation-based End-to-End Learning Framework for Evidential Occupancy Grid Mapping* ([arXiv 2021](https://arxiv.org/abs/2102.12718)) | E2E evidential OGM | 比较早的“端到端占据图 evidential”路线。 |
| *Evidential Occupancy Grid Map Augmentation using Deep Learning* ([arXiv 2018](https://arxiv.org/abs/1801.05297)) | 占据图增强 | 早期证据占据图思路。 |
| *Uncertainty Estimation for 3D Object Detection via Evidential Learning* ([arXiv 2024](https://arxiv.org/abs/2410.23910)) | 3D 检测不确定性 | 处于 preprint 阶段，显示 EDL 正在往 3D 检测渗透。 |
| *MEDL-U: Uncertainty-aware 3D Automatic Annotation based on EDL* ([arXiv 2023](https://arxiv.org/abs/2309.09599)) | 3D 自动标注/伪标签 | 说明 EDL 也被用于“数据层”改进，不只模型推理层。 |

小结：自动驾驶里 EDL 最“扎实”的落地点是感知/占据/地图侧。

---

## 4.2 预测 / 规划 / 控制（正在形成）

| 文献 | 级别 | 任务 | 与你路线关系 |
|---|---|---|---|
| Itkina et al., *Interpretable Self-Aware Neural Networks for Robust Trajectory Prediction* ([CoRL 2022 / PMLR 2023](https://proceedings.mlr.press/v205/itkina23a.html)) | A | 轨迹预测 | 说明“self-aware/evidential”在 driving trajectory prediction 已有正式工作。 |
| *Evidential Uncertainty Estimation for Multi-Modal Trajectory Prediction* ([arXiv 2025](https://arxiv.org/abs/2503.05274)) | C | 多模态轨迹预测 | 直接对标你的“模态不确定性”方向，但仍是 preprint。 |
| *RuleFuser: An Evidential Bayes Approach for Rule Injection in Imitation Learned Planners and Predictors* ([arXiv 2024](https://arxiv.org/abs/2405.11139)) | C | IL planner/predictor | 与你最相关：把 evidential Bayes 注入 imitation planner，用于 OOD 与规则约束。 |
| *DRO-EDL-MPC* ([arXiv 2025](https://arxiv.org/abs/2507.05710), [OpenReview](https://openreview.net/forum?id=JigwaMzv4c)) | C | 感知不确定性 + 鲁棒 MPC | 说明 EDL 已开始进入控制层，但仍偏 early-stage。 |
| *EVORA: Deep Evidential Traversability Learning for Risk-Aware Off-Road Autonomy* ([arXiv 2023](https://arxiv.org/abs/2311.06234)) | C | 越野风险感知与规划 | 证明“证据不确定性驱动风险规避”在机器人规划有效。 |

小结：预测/规划/控制侧已有明显尝试，但高质量、统一 benchmark 的闭环证据还不够。

---

## 4.3 端到端驾驶主线（TransFuser / Diffusion / RL）上的现状

相关基线：
- TransFuser ([arXiv](https://arxiv.org/abs/2205.15997))
- Diffusion-ES（nuPlan 规划）([arXiv](https://arxiv.org/abs/2402.06559))
- Diffusion Policy（机器人策略）([arXiv](https://arxiv.org/abs/2303.04137))
- UncAD（端到端在线地图不确定性）([arXiv](https://arxiv.org/abs/2504.12826))

判定（基于本次检索）
- 结论是“部分做过，但没有做透到你想要的完整形态”。
- 已做过：EDL 在感知/占据/预测、以及少量 planner/control 里已有工作。
- 尚缺口：
  1. 将 Dirichlet/EDL 直接深度耦合到 `TransFuser/扩散规划器` 的主训练目标并在标准闭环 benchmark（如 nuPlan/NAVSIM）系统验证。
  2. 将 evidential 信号与 RL 微调（策略改进）联合优化的成体系研究。
  3. 在“安全收益 vs 效率损失”上给出跨场景、跨 OOD 分桶的严格统计结论。

这是一个“有前人、有空白”的课题，不是从零，也不是红海饱和。

---

## 5. 对你这条路线的学术判断

## 5.1 你现在的想法是否新颖
你的核心主张是：
- 不只追求更小轨迹误差，而是用 evidential 信号减少高风险和 OOD 场景的大错，并支持可控降级。

这个主张与已有工作一致的部分：
- 风险/不确定性驱动决策在文献中成立。

仍有创新空间的部分（更可能出论文）：
1. **方法层创新**：把 EDL uncertainty 直接作用到 planner 的动作分布/轨迹打分，而非只做后处理门控。
2. **训练层创新**：IL + RL 联合时，让 evidential 信号进入奖励或约束（风险敏感微调）。
3. **评测层创新**：把 OOD 分桶（长尾交通参与者、遮挡、稀有规则冲突）下的 safety-efficiency frontier 做系统比较。

## 5.2 你的主要风险
1. EDL 可能出现“证据塌缩/伪校准”，需要严格校准评估与 ablation（TPAMI 2025 已提示该风险）。
2. 仅靠一个 uncertainty 指标很难说服审稿人，需要“行为层收益”闭环指标联动证明。
3. 若只在 toy 场景有效，贡献会被认为偏工程 trick。

---

## 6. 建议的论文叙事（可直接用于开题/组会）

主问题：
- Can evidential uncertainty be turned from a diagnostic signal into a train-time control signal for safer planning under distribution shift?

核心贡献目标：
1. 一个可插拔的 evidential planning head（兼容 TransFuser/扩散规划器）。
2. 一个 uncertainty-aware 训练机制（IL + 可选 RL 微调）。
3. 一套风险导向评测协议：
   - 安全：collision / TTC violation / hard-brake proxy
   - 效率：progress / route completion / comfort
   - 可靠性：ECE / NLL / OOD-AUROC / failure calibration curve

如果三者同时成立，你的工作会比“仅提升 ADE/FDE”更有说服力。

---

## 7. 一句话结论
- **“EDL 用于自动驾驶训练”这件事已经被做过，但主要集中在感知/占据与部分预测；把 EDL 深度融入端到端规划（尤其结合 RL 微调）并在闭环基准上形成完整证据链，仍有明显研究空白。**

---

## 8. 参考链接（本次综述使用）

### EDL 基础
- https://proceedings.neurips.cc/paper_files/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html
- https://arxiv.org/abs/1806.01768
- https://proceedings.neurips.cc/paper_files/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html
- https://arxiv.org/abs/1910.02600
- https://papers.neurips.cc/paper_files/paper/2018/hash/3ea2db50e62ceefceaf70f689bf8f9f3-Abstract.html
- https://arxiv.org/abs/1802.10501
- https://arxiv.org/abs/2409.04720
- https://pubmed.ncbi.nlm.nih.gov/40569804/

### 自动驾驶相关
- https://openaccess.thecvf.com/content/CVPR2024/papers/Kalble_Accurate_Training_Data_for_Occupancy_Map_Prediction_in_Automated_Driving_CVPR_2024_paper.pdf
- https://openaccess.thecvf.com/content/CVPR2025/papers/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.pdf
- https://arxiv.org/abs/2102.12718
- https://arxiv.org/abs/1801.05297
- https://arxiv.org/abs/2410.23910
- https://arxiv.org/abs/2309.09599
- https://proceedings.mlr.press/v205/itkina23a.html
- https://arxiv.org/abs/2503.05274
- https://arxiv.org/abs/2405.11139
- https://arxiv.org/abs/2507.05710
- https://openreview.net/forum?id=JigwaMzv4c
- https://arxiv.org/abs/2311.06234

### 端到端/扩散/不确定性对照
- https://arxiv.org/abs/2205.15997
- https://arxiv.org/abs/2402.06559
- https://arxiv.org/abs/2303.04137
- https://arxiv.org/abs/2504.12826
