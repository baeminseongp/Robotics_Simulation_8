# Evaluation Results

이 디렉터리는 학습 log가 아닌 최종 policy evaluation의 원본 결과를 보관한다.

## 공식 결과

| 결과 파일 | Model | Suite | Episodes | Mean | Sample std |
| --- | --- | --- | ---: | ---: | ---: |
| [`curriculum_seed0_combined.json`](curriculum_seed0_combined.json) | Curriculum seed 0, iteration 999 | Combined | 100 | **29.118270** | **12.040080** |

## 평가 조건

- Checkpoint: `../checkpoints/curriculum_seed0_model_999.pt`
- Evaluation seed: `30000`
- Parallel environments: `100`
- Episodes: environment당 첫 episode 1개
- Policy: deterministic inference
- Online gradient update: 없음
- Combined suite: held-out rough terrain + contact friction `0.2`

JSON의 `episode_returns`에는 100개 environment의 개별 cumulative return이 들어 있다. `episode_return_mean`은 이 값들의 평균, `episode_return_std`는 sample standard deviation이다.

실행 명령과 전체 해석은 [`../README.md`](../README.md)와 [`../reports/RESULTS.md`](../reports/RESULTS.md)를 참고한다.
