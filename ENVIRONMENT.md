# Reproduction Environment

| 항목 | 값 |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS |
| GPU | NVIDIA GeForce RTX 3090 24 GB |
| Driver | 580.178.04 |
| Isaac Lab commit | `f4aa17f87e2e5db5484f0b5974918573e8918ce2` |
| Isaac Lab describe | `v2.3.2-13-gf4aa17f87e2` |
| Python | 3.11 |
| PyTorch | `2.7.0+cu128` |
| rsl-rl-lib | `5.0.1` |
| Device | `cuda:0` |

학습 source와 실제 run config가 달라지는 것을 방지하기 위해 각 log directory의 `params/env.yaml`과 `params/agent.yaml`을 함께 제출했다.

