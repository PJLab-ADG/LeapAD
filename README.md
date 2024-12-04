# <img src="assets/figures/leap.png" alt="icon" style="width: 40px; height: 40px; vertical-align: middle;"> Continuously <span style="color:#ff7e5f">Le</span>arning, <span style="color:#ff7e5f">A</span>dapting, and Im<span style="color:#ff7e5f">p</span>roving: A Dual-Process Approach to <span style="color:#ff7e5f">A</span>utonomous <span style="color:#ff7e5f">D</span>riving

[![Custom badge](https://img.shields.io/badge/Arxiv-pdf-8A2BE2?logo=arxiv)](https://arxiv.org/abs/2405.15324) [![Custom badge](https://img.shields.io/badge/Project-page-green?logo=document)](https://pjlab-adg.github.io/LeapAD/) ![GitHub license](https://img.shields.io/badge/License-Apache--2.0-red)

<!-- **<span style="color:#ff7e5f">LeapAD</span>**, a new autonomous driving paradigm inspired by human cognition, improves adaptability and interpretability in complex scenarios through dual-process decision-making and continuous learning from past experiences. -->


> Jianbiao Mei<sup>1,2,\*</sup>, Yukai Ma<sup>1,2,\*</sup>, Xuemeng Yang<sup>2</sup>, Licheng Wen<sup>2</sup>, Xinyu Cai<sup>2</sup>, Xin Li<sup>2,4</sup>, Daocheng Fu<sup>2</sup>, Bo Zhang<sup>2</sup>, Pinlong Cai<sup>2</sup>, Min Dou<sup>2</sup>, Botian Shi<sup>2,†</sup>, Liang He<sup>3</sup>, Yong Liu<sup>1,†</sup>, Yu Qiao<sup>2</sup> <br>
> <sup>1</sup> Zhejiang University <sup>2</sup> Shanghai Artificial Intelligence Laboratory <sup>3</sup> East China Normal University <sup>4</sup> Shanghai Jiao Tong University<br>
> <sup>\*</sup> Equal Contribution <sup>†</sup> Corresponding Authors

## 📖 News
- `[2024-9-26]` Our LeapAD is accepted by NeurIPS 2024!

- `[2024-5-27]` The paper can be accessed at [arxiv](https://arxiv.org/abs/2405.15324).

- `[2024-5-22]` We released our project website [here](https://pjlab-adg.github.io/LeapAD/).

---

## 🎯 Overview
We introduce **<span style="color:#ff7e5f">LeapAD</span>**, a novel paradigm for autonomous driving inspired by the human cognitive process. Specifically, LeapAD emulates human attention by selecting critical objects relevant to driving decisions, simplifying environmental interpretation, and mitigating decision-making complexities. Additionally, LeapAD incorporates an innovative dual-process decision-making module, which consists of an **Analytic Process** (System-II) for thorough analysis and reasoning, along with a **Heuristic Process** (System-I) for swift and empirical processing. 

<div style="text-align:center;">
  <img src="assets/figures/brief_pipeline.png" alt="pipeline" width="600">
</div>

The <span style="color:#B46504">scene understanding module</span> analyzes surrounding images and provides descriptions of critical objects that may influence driving decisions. These scenario descriptions are then fed into the <span style="color:#EA6B66">dual-process decision module</span> for reasoning and decision-making. The generated decisions are then transmitted to <span style="color:#A680B8">action executor</span>, where they are converted into control signals for interaction with the <span style="color: #1BA1E2">simulator</span>. 
The Analytic Process then uses an LLM to accumulate driving analysis and decision-making experience and conduct reflection on accidents. The experience is stored in the <span style="color:#009600">memory bank</span> and transferred to a lightweight language model, forming our Heuristic Process for quick responses and continuous learning.

<div style="text-align:center;">
    <img src="assets/figures/reflection.png" alt="pipeline" width="600">
</div>

When Heuristic Process encounters traffic accidents, the Analytic Process intervenes, analyzing historical frames to pinpoint errors and provide corrected samples. These corrected samples are then integrated into the memory bank to facilitate continuous learning.

<!-- The **Analytic Process** is designed for thorough analysis and reasoning. It handles complex scenarios and builds a comprehensive memory bank for high-quality driving decisions. The Analytic Process accumulates experience and updates the memory bank through analysis of accidents and self-reflection. This accumulated knowledge can be transferred into the Heuristic Process by supervised fine-tuning (SFT), ensuring the entire LeapAD system can continuously improve and adapt to new driving environments and challenges.

The **Heuristic Process** uses several strategies to perform closed-loop decisions.  It is designed to enable instant decision-making within the vehicle. The Heuristic Process relies on knowledge transferred from the analytical process to make fast and efficient decisions during driving. This lightweight model ensures fast response and adaptability in various driving scenarios, maintaining a high level of performance with minimal computing resources. -->

## 🤩 Getting Started

* Please follow [INSTALL.md](docs/INSTALL.md) to setup the environment, make sure you are using a machine with **at least** a mid-end GPU to run CARLA.
*  To run the LeapAD within CARLA, please refer to the [QUICK_START.md](docs/QUICK_START.md) .

## 🛣️ Demo Video in CARLA

https://github.com/PJLab-ADG/LeapAD/assets/18390668/f5383343-a5cd-4fd3-aaf7-98302d9ea6cf

https://github.com/user-attachments/assets/afcc2002-23ed-41ec-9ca3-f3b286f26bc7

https://github.com/PJLab-ADG/LeapAD/assets/18390668/a4dd470f-c1ef-4e55-8537-ded2ae8101d6


We conduct closed-loop tests in CARLA. It can be seen that LeapAD can make informed decisions using the Heuristic Process with only 1.8B parameters while driving. 
<!-- Experiments show that LeapAD outperforms all methods that rely solely on camera input, requiring 1-2 orders of magnitude less annotated data. As the memory base expands, Heuristic Process with only 1.8B parameters can inherit the knowledge of GPT-4 powered Analytic Process and achieve continuous performance improvements. -->

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

We utilized the following repos during development:

* [carla-leaderboard](https://github.com/carla-simulator/leaderboard)
* [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
* [Qwen1.5](https://github.com/QwenLM/Qwen2.5/tree/v1.5)

Thanks for their Awesome open-sourced work!

## 📄 License

This project is released under the [Apache 2.0 license](LICENSE). 

## 🔖 Citation

If you find our project useful, please kindly cite us via:

```bibtext
@article{mei2024continuously,
          title={Continuously Learning, Adapting, and Improving: A Dual-Process Approach to Autonomous Driving},
          author={Mei, Jianbiao and Ma, Yukai and Yang, Xuemeng and Wen, Licheng and Cai, Xinyu and Li, Xin and Fu, Daocheng and Zhang, Bo and Cai, Pinlong and Dou, Min and others},
          journal={arXiv preprint arXiv:2405.15324}
