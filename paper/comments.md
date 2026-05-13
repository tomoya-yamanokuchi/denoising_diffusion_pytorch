# Reviewer Comments — APIN-D-26-02746

**Manuscript**: VoxelDiffusionCut: Non-destructive Internal-part Extraction via Iterative Cutting and Structure Estimation
**Journal**: Applied Intelligence
**Decision**: Major Revision
**Authors**: Takumi Hachimine, Yuhwan Kwon, Cheng-Yu Kuo, Tomoya Yamanokuchi (corresponding), Takamitsu Matsubara

---

## 0. Editor Note

> AE: One of the received reviews is not applicable to the submitted manuscript and has therefore not been considered.

→ 依 AE 的判斷，**Reviewer #1** 與本論文無關，未被列入考量。下方仍保留其原文於附錄存檔。

---

## 1. Associate Editor (AE) — 原文

> AE: One of the received reviews is not applicable to the submitted manuscript and has therefore not been considered. Based on the remaining relevant review and my assessment, the manuscript presents a potentially interesting framework for non-destructive internal-part extraction using voxel-based conditional diffusion and iterative cutting planning. However, the current version requires major revision before it can be considered further. The authors should substantially strengthen the justification of key parameters, including the cutting-risk threshold, guidance scale, sampling settings, and voxel resolution; clarify the convergence and reliability of the iterative estimation–planning loop; improve the fairness and transparency of the baseline comparisons; and provide stronger evidence regarding scalability and deployment feasibility under more realistic sensing and cutting conditions. The revised manuscript should also moderate claims, given that the current validation is entirely simulation-based and assumes ideal cutting execution and noise-free observations.

---

## 2. Reviewer #3 — 原文（逐句拆分編號，未改寫文字）

下列 R3-1 ~ R3-10 直接對應 reviewer 原文的句子，僅做斷句與編號，未改寫文字。

> The significant apprehensions emerge as the framework is overlooked regarding real environment constraints where noise affects cutting accuracy severely causing unreliable deployment scenarios across diverse dismantling conditions.

**R3-1**: *The significant apprehensions emerge as the framework is overlooked regarding real environment constraints where noise affects cutting accuracy severely causing unreliable deployment scenarios across diverse dismantling conditions.*

---

> Equally, an VoxelDiffusionCut reveals ambiguous parameter behavior since tuning strategy is unjustified clearly reducing interpretability of diffusion behavior among varying internal structure estimation tasks during execution stages.

**R3-2**: *Equally, an VoxelDiffusionCut reveals ambiguous parameter behavior since tuning strategy is unjustified clearly reducing interpretability of diffusion behavior among varying internal structure estimation tasks during execution stages.*

---

> Besides, a reliability becomes questionable because iterative estimation remains unclear due to missing convergence explanation which weakens confidence in sequential updates modifying planning reliability using conditional diffusion modeling approach.

**R3-3**: *Besides, a reliability becomes questionable because iterative estimation remains unclear due to missing convergence explanation which weakens confidence in sequential updates modifying planning reliability using conditional diffusion modeling approach.*

---

> In addition, the complex geometry handling through voxelization with part attribute encoding appears limited when representing fine structures leading to coarse approximations that reduce structural precision in internal component boundaries significantly.

**R3-4**: *In addition, the complex geometry handling through voxelization with part attribute encoding appears limited when representing fine structures leading to coarse approximations that reduce structural precision in internal component boundaries significantly.*

---

> On top of that, a Classifier-Free Guidance presents baseless configuration as guidance scaling influence is unevaluated systematically leading to uncertain conditioning strength across different observation scenarios within generative prediction workflow.

**R3-5**: *On top of that, a Classifier-Free Guidance presents baseless configuration as guidance scaling influence is unevaluated systematically leading to uncertain conditioning strength across different observation scenarios within generative prediction workflow.*

---

> On the other hand, the performance interpretation becomes deficient because baseline configurations are not consistently aligned causing unfair comparison among methods affecting credibility of conclusions derived using VAEAC modeling approach.

**R3-6**: *On the other hand, the performance interpretation becomes deficient because baseline configurations are not consistently aligned causing unfair comparison among methods affecting credibility of conclusions derived using VAEAC modeling approach.*

---

> Congruently, a broader applicability damages since data generation using synthetic structural distributions is minimal which restricts exposure to real variations about unseen product configurations during deployment stages significantly.

**R3-7**: *Congruently, a broader applicability damages since data generation using synthetic structural distributions is minimal which restricts exposure to real variations about unseen product configurations during deployment stages significantly.*

---

> Notably, an DDIM sampling shows obscure computational characteristics as scalability for higher resolution voxel grids remains unexplored creating improbability in real time applicability within industrial dismantling pipelines environments.

**R3-8**: *Notably, an DDIM sampling shows obscure computational characteristics as scalability for higher resolution voxel grids remains unexplored creating improbability in real time applicability within industrial dismantling pipelines environments.*

---

> Afterwards, a practical deployment becomes challenging because feature encoding is unreasonable due to reliance on simplified color attributes ignoring realistic sensing complexities decreasing applicability using voxel feature encoding in deployment scenarios.

**R3-9**: *Afterwards, a practical deployment becomes challenging because feature encoding is unreasonable due to reliance on simplified color attributes ignoring realistic sensing complexities decreasing applicability using voxel feature encoding in deployment scenarios.*

---

> Thereafter, an evaluation clarity diminishes since assessment using cutting action planning formulation was vague where safety critical factors are unexamined thoroughly weakening effectiveness across inexact operational conditions during execution phases.

**R3-10**: *Thereafter, an evaluation clarity diminishes since assessment using cutting action planning formulation was vague where safety critical factors are unexamined thoroughly weakening effectiveness across inexact operational conditions during execution phases.*

---

