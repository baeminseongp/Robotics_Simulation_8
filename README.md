# Isaac-Ant Unseen Terrain Generalization

Isaac Lab의 `Isaac-Ant-v0`를 다양한 terrain geometry와 contact friction에서 학습해 unseen terrain의 robustness를 높인 PPO 프로젝트다. 원본 reward와 PPO hyperparameter는 유지하고, training distribution만 단계적으로 확장했다.

## 최종 평가 결과

2026-10-02에 `Curriculum` seed 0의 최종 checkpoint를 `combined` suite에서 평가했다. 100개 parallel environment에서 각각 첫 episode 하나만 수집한 결과다.

| Model | Suite | Episodes | Episode return mean | Episode return std |
| --- | --- | ---: | ---: | ---: |
| Curriculum seed 0 (`model_999.pt`) | Combined | 100 | **29.118270** | **12.040080** |

- 원본 결과: [`results/curriculum_seed0_combined.json`](results/curriculum_seed0_combined.json)
- 평가 조건과 해석: [`reports/RESULTS.md`](reports/RESULTS.md)
- 결과 파일 안내: [`results/README.md`](results/README.md)

## 제출물 구성

| 요구 항목 | 위치 |
| --- | --- |
| Environment code | `source/isaaclab_tasks/.../ant/generalization/` |
| RL / training code | `experiments/ant_generalization/` |
| 실행 당시 config | `configs/`, 각 run의 `params/` |
| TensorBoard log | `logs/rsl_rl/ant_generalization_*` |
| Checkpoint | `checkpoints/curriculum_seed0_model_999.pt` 및 각 run의 `model_999.pt` |
| 평가 명령어 | 아래 **평가** 절 |
| 평가 원본 결과 | `results/curriculum_seed0_combined.json` |
| 평가 결과와 해석 | `reports/RESULTS.md` |
| Training curve | `reports/training_return_curve.png` |

`ant_generalization_smoke`의 `model_1.pt`와 smoke log는 제출 결과에서 완전히 제외했다. 저장소의 checkpoint는 모두 1,000 iteration 본 학습에서 생성된 `model_999.pt`다.

## 실험 설계 및 방법

실제 운용 시 지형 형상과 접촉 마찰이 달라질 수 있는 다양한 환경을 가정해 실험을 설계했다. 기본 PPO에 하나의 요소씩 누적해 비교하는 **ablation study**를 수행하고, Combined 환경의 robustness와 seed 안정성을 기준으로 최적 방법을 선정했다. 공정한 비교를 위해 원본 reward, PPO hyperparameter, 학습 budget을 고정하고 각 variant를 seed `0, 1, 2`로 학습했다.

누적 ablation은 다음과 같다.

1. `PPO`: 원본 flat environment
2. `Terrain DR`: Flat 20% + Random Noise 30% + Random Blocks 30% + Slope 20%
3. `Friction DR`: terrain + environment별 contact friction + 작은 initial-state perturbation
4. `Curriculum`: 전진 거리에 따라 terrain level과 friction range를 함께 조정
5. `4-frame History`: curriculum + 최근 observation 4개 frame

최종 model은 Combined robustness와 seed 안정성이 가장 좋은 `Curriculum`의 seed 0 checkpoint다.

### 중요 파라미터

| 구분 | 설정 | 원본 파일 |
| --- | --- | --- |
| 학습 규모 | `4096 env × 32 step × 1000 iteration` = seed당 `131,072,000 transition` | [`configs/curriculum_seed0_agent.yaml`](configs/curriculum_seed0_agent.yaml), [`configs/curriculum_seed0_env.yaml`](configs/curriculum_seed0_env.yaml) |
| Policy / PPO | Actor·Critic MLP `400-200-100`, ELU; learning rate `5e-4` (adaptive), `gamma=0.99`, `lambda=0.95`, clip `0.2`, epoch `5`, mini-batch `4` | [`configs/curriculum_seed0_agent.yaml`](configs/curriculum_seed0_agent.yaml) |
| Simulation | physics `dt=1/120 s`, decimation `2`, episode `16 s` | [`configs/curriculum_seed0_env.yaml`](configs/curriculum_seed0_env.yaml) |
| Terrain DR | Flat/Noise/Blocks/Slope = `20/30/30/20%`; 학습 roughness 최대 `0.10 m`, slope 최대 `0.35` | [`ant_generalization_env_cfg.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/generalization/ant_generalization_env_cfg.py) |
| Friction curriculum | terrain level에 따라 `0.7–1.3 → 0.5–1.5 → 0.4–1.6 → 0.3–1.7`; +x 전진 `>3.0 m` 승격, `<0.75 m` 강등 | [`ant_generalization_env_cfg.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/generalization/ant_generalization_env_cfg.py), [`mdp.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/generalization/mdp.py) |
| 최종 평가 | seed `30000`, `100 env`, environment당 첫 episode 1개, deterministic policy; unseen roughness `0.08–0.12 m` + friction `0.2` | [`play_one_episode.py`](experiments/ant_generalization/play_one_episode.py), [`ant_generalization_env_cfg.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/generalization/ant_generalization_env_cfg.py) |

## 실행 환경

- Ubuntu 24.04
- NVIDIA RTX 3090, driver 580.178.04
- Isaac Lab commit `f4aa17f87e2e5db5484f0b5974918573e8918ce2` (`v2.3.2-13-gf4aa17f87e2`)
- Python 3.11
- PyTorch `2.7.0+cu128`
- `rsl-rl-lib 5.0.1`

이 저장소는 전체 Isaac Lab 복사본이 아니라 과제 관련 파일만 포함하는 overlay다. 위 commit의 Isaac Lab root에 같은 상대 경로로 파일을 복사한 뒤 실행한다.

## 학습

Isaac Lab root에서 다음 명령을 실행한다.

```bash
setup-image/.venv/bin/python \
  experiments/ant_generalization/prepare.py \
  --variant curriculum --seed 0 --execute
```

전체 5 variant × 3 seed:

```bash
setup-image/.venv/bin/python \
  experiments/ant_generalization/train_matrix.py --execute
```

본 학습은 run당 `4096 env × 32 step × 1000 iteration = 131,072,000 transition`이다.

## 평가

과제 평가 결과는 반드시 `play_one_episode.py`의 산출물을 사용한다. Script는 100개 parallel environment에서 각각 첫 episode 하나를 수집한 뒤, environment별 cumulative episode return의 mean과 sample standard deviation을 자동 출력한다.

Isaac Lab root에서 실행:

```bash
export PYTHONPATH="$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl${PYTHONPATH:+:$PYTHONPATH}"

setup-image/.venv/bin/python \
  experiments/ant_generalization/play_one_episode.py \
  --headless --device cuda:0 \
  --checkpoint checkpoints/curriculum_seed0_model_999.pt \
  --suite combined \
  --num_envs 100 \
  --eval_seed 30000 \
  --output results/curriculum_seed0_combined.json
```

`combined` suite는 held-out rough terrain과 고정 contact friction `0.2`를 동시에 적용한다. 다른 suite는 `flat`, `seen`, `unseen_rough`, `unseen_slippery` 중 선택할 수 있다.

## Log 확인

```bash
setup-image/.venv/bin/python -m tensorboard.main \
  --logdir logs/rsl_rl --host 127.0.0.1 --port 6006
```

Training curve를 다시 생성하려면:

```bash
setup-image/.venv/bin/python reports/plot_training_curves.py
```

![Training return curve](reports/training_return_curve.png)

## 주요 파일

- `ant_generalization_env_cfg.py`: terrain, friction, curriculum, evaluation suite
- `mdp.py`: custom terrain generator, terrain curriculum, foot material randomization
- `rsl_rl_ppo_cfg.py`: PPO runner config
- `prepare.py`: variant별 재현 가능한 training command
- `play_one_episode.py`: 공식 100-environment 평가 script
- `reports/RESULTS.md`: 평가 결과 provenance와 해석
