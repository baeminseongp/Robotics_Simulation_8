# Experiment Design

## Task

기본 `Isaac-Ant-v0` policy가 학습에서 보지 못한 terrain geometry와 낮은 contact friction에서도 전진하도록 만드는 것이 목표다. 평가 reward를 다시 설계하지 않고 training distribution만 변경해, 성능 차이가 reward shaping에서 발생하지 않도록 했다.

## Hypothesis

1. `Terrain DR`은 rough geometry에 대한 generalization을 높일 것이다.
2. `Friction DR`은 slippery contact가 추가된 환경의 성능 저하를 줄일 것이다.
3. `Performance-based Curriculum`은 처음부터 전체 난이도를 제시하는 것보다 안정적으로 어려운 terrain–friction 조합을 학습하게 할 것이다.
4. `4-frame History`는 recent dynamics를 제공하지만, 단순 frame stacking만으로는 모든 조건에서 일관된 추가 이득이 없을 수 있다.

## Controlled Comparison

모든 variant에서 다음 항목을 고정했다.

- Robot USD, mass, inertia, actuator
- Action scale, control frequency, physics timestep
- Reward term과 weight
- Termination과 episode duration
- PPO network와 optimizer hyperparameter
- Training transition budget
- Training seed `0, 1, 2`

변경한 항목은 terrain distribution, contact friction, 작은 initial-state perturbation, curriculum, observation history뿐이다.

## Training Variants

| Variant | 추가되는 요소 |
| --- | --- |
| PPO | 원본 flat PPO |
| Terrain | 20/30/30/20 terrain mixture |
| Friction | reset별 foot material friction, initial-state perturbation |
| Curriculum | forward distance 기반 terrain level, level별 friction range |
| History | policy observation 4-frame stacking |

각 run은 `4096 env × 32 step × 1000 iteration`으로 동일한 131,072,000 transition을 사용했다.

## Curriculum

- Episode +x displacement가 `3.0 m`보다 크면 terrain level을 올린다.
- `0.75 m`보다 작으면 terrain level을 내린다.
- Friction range는 level에 따라 `0.7–1.3`, `0.5–1.5`, `0.4–1.6`, `0.3–1.7`로 확대한다.

공유 terrain mesh에서는 ground material을 environment별로 바꿀 수 없으므로, 네 foot의 material을 environment별로 sampling하고 ground의 `multiply` combine mode와 결합했다.

## Train/Test Separation

| 구분 | Terrain seed | 조건 |
| --- | ---: | --- |
| Training | 10000 | 최대 10 cm, friction 0.3–1.7 |
| Seen evaluation | 20000 | Training family와 범위, 별도 mesh |
| Unseen evaluation | 30000 | 8–12 cm Noise/Blocks, 30 cm block width |
| Combined | 30000 | Unseen terrain + friction 0.2 |

Test return으로 checkpoint나 hyperparameter를 선택하지 않았다.

## Official Assignment Evaluation

과제의 공식 결과는 `play_one_episode.py --num_envs 100`이 출력하는 episode return mean/std다. 100개 environment의 첫 episode만 사용하며 deterministic inference 중에는 parameter update를 수행하지 않는다.

