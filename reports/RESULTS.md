# Evaluation Result

## 평가 규칙

- 평가 script: `experiments/ant_generalization/play_one_episode.py`
- Model: `Curriculum`, training seed 0
- Checkpoint: `model_999.pt`
- Suite: `combined`
- Evaluation seed: `30000`
- Parallel environments: `100`
- Environment당 episode: 첫 episode 1개
- Policy: deterministic inference, evaluation 중 update 없음
- 집계: 100개 cumulative episode return의 mean ± sample standard deviation

> 이 문서의 공식 수치는 기존 `evaluate.py` CSV나 smoke run의 산출물이 아니다. 위 `play_one_episode.py`를 실제로 실행해 생성한 JSON만 사용한다.

## 결과

실행 중이며 완료된 JSON 값을 이 표에 기록한다.

| Suite | Episodes | Episode return mean | Episode return std |
| --- | ---: | ---: | ---: |
| Combined | 100 | pending | pending |

## 해석

Combined는 held-out rough terrain과 training 범위 밖의 낮은 friction을 동시에 적용한 가장 어려운 조건이다. 따라서 이 결과는 flat locomotion 속도가 아니라 geometry와 dynamics 변화가 겹칠 때의 robustness를 나타낸다.

높은 return만으로 method를 평가하지 않기 위해 다음 사항을 함께 통제했다.

- 원본 Ant reward와 PPO hyperparameter 유지
- Training/test terrain seed 분리
- 100개 environment를 동일한 evaluation seed로 평가
- Smoke checkpoint 제외
- Evaluation 중 parameter update 금지

