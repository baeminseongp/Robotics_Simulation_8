# Results — 무엇이 일반화되었고, 어디서 실패했는가?

[프로젝트 요약](../README.md) · [방법과 가설](METHOD.md) · [학습 비교](#ablation) · [팀 평가](#evaluation) · [가설 검증](#hypotheses) · [원본 자료](#artifacts)

**Curriculum seed 0는 새로운 rails·gap에서 Seen-Control에 가까운 평균 return을 보였지만, cylinders와 넓은 마찰 범위가 결합된 Hard에서는 생존율이 14%로 낮아졌다.** 현재 결과는 이 모델의 전이 범위와 한계를 보여준다. 다른 variant의 동일 조건 평가가 없어 DR·Curriculum 각각의 일반화 개선량은 확정할 수 없다.

<a id="ablation"></a>
## 1. 학습 중 누적 ablation

15개 TensorBoard 로그의 `Train/mean_reward`를 사용했다. 각 seed에서 **iteration 900–999의 100개 기록을 평균**한 뒤, 3개 seed 평균의 mean ± sample std(`ddof=1`)를 계산했다. 평가 episode 통계와 다른 지표다.

| Variant | 추가되는 요소 | Seed 0 | Seed 1 | Seed 2 | 3-seed 평균 ± std |
| --- | --- | ---: | ---: | ---: | ---: |
| PPO | 원본 flat | 117.368 | 108.460 | 109.639 | **111.822 ± 4.839** |
| Terrain | 지형 혼합 | 40.707 | 41.403 | 41.772 | **41.294 ± 0.541** |
| Friction | 마찰 + 초기 상태 변화 | 42.356 | 43.081 | 41.594 | **42.344 ± 0.743** |
| Curriculum | 성과 기반 난이도 + 범위 확대 | 41.049 | 39.295 | 40.289 | **40.211 ± 0.879** |
| History | 최근 관측 4개 연결 | 39.932 | 39.872 | 40.775 | **40.193 ± 0.505** |

[집계 CSV](training_ablation.csv) · [집계 코드](summarize_training.py) · [원본 로그](../logs/rsl_rl/)

![학습 return: 얇은 선은 개별 seed, 굵은 선과 음영은 3-seed 평균과 표준편차](training_return_curve.png)

### 이 비교에서 읽을 수 있는 것

- **PPO → Terrain:** return 하락은 평지와 혼합 지형의 학습 난도가 다르다는 사실과 함께 읽어야 한다. PPO의 높은 평지 return은 미지 지형에서도 우수하다는 증거가 아니다.
- **Terrain → Friction:** 마지막 구간 평균이 1.050 높다. 마찰·초기 상태 변화를 추가하고도 해당 학습 분포에서 비슷한 수준의 return에 도달했다. 낮은 마찰에서의 개선량이나 통계적으로 유의한 우위는 측정하지 않았다.
- **Friction → Curriculum:** 평균은 2.133 낮다. Curriculum은 더 넓은 지형·마찰 범위로 이동하므로 이 차이만으로 학습을 방해했다고 판단할 수 없다. 반대로 성능 안정화에 성공했다고 단정할 수도 없다. 동일한 최종 분포를 사용하는 non-curriculum 대조군이 필요하다.
- **Curriculum → History:** 평균 차이는 −0.018로 작다. seed 간 std는 0.879에서 0.505로 줄지만 seed 3개로 안정성 개선을 확정할 수 없다. 이 학습 지표에서 뚜렷한 추가 이득은 관찰되지 않았으며, 미지 접촉에 대한 도움은 별도 평가가 필요하다.

**설계의 범위:** 누적 구성 비교이며 독립 요인 실험이 아니다. Friction에는 reset 변화, Curriculum에는 범위·마찰 분포 변화, History에는 입력층 용량 증가가 동반된다. [실제 설정 차이](METHOD.md#3-구현과-누적-비교)

<a id="evaluation"></a>
## 2. 최신 팀 공통 평가 v2.1

### 평가 규칙

| 항목 | 적용 규칙 |
| --- | --- |
| 실행 | 2026-10-05, run `20261005_195743` (Asia/Seoul) |
| 모델 | Curriculum, training seed 0, `model_999.pt`; 네 환경에서 같은 checkpoint |
| CLI seed / 지형 seed | CLI seed **24**; 환경별 별도 고정 mesh seed는 아래 표 |
| 표본 | 환경별 parallel env 100개에서 **첫 episode 1개씩**, 총 400 episodes |
| 추론 | deterministic; gradient update 없음; 학습 curriculum manager 비활성화 |
| 입력·출력 | 관측 60, 행동 8; 학습의 관측 순서·action scale 유지 |
| Episode | 최대 960 policy steps(16초); torso world 높이 <0.31 m이면 조기 종료 |
| Return | 원본 7항 가중 reward × step_dt의 episode 누적값; 조기 종료도 포함 |
| 표준편차 | 100개 episode return의 sample std(`ddof=1`); 표준오차·신뢰구간 아님 |
| 보조 지표 | episode steps, reset 직후부터 종료 직전까지 +x 변위, 960-step 도달 비율 |
| 완료 기준 | `Completed first episodes: 100/100`; **수집 완료율과 생존율은 다름** |

학습의 초기 상태 perturbation은 유지했다. 팀 평가의 마찰은 startup 때 **robot의 모든 collision shape**에 환경별 동일 재질을 적용한다. 학습은 reset 때 발만 바꾸므로 접촉 재질 적용 범위에도 차이가 있다. ground는 마찰 1.0, `multiply`다. 정지·동마찰을 샘플링하고 동마찰 ≤ 정지마찰을 유지한다.

### 환경별 평가 의도

| 환경 | 지형 seed / 구성 | 정지 / 동마찰 범위 | 평가하는 변화 |
| --- | --- | --- | --- |
| Seen-Control | 51004; plane/noise/grid/slope 각 25%, d=0.45 | 0.50–1.50 / 0.40–1.20 | 익숙한 family의 기준 조건; 학습 분포와 완전히 같지는 않음 |
| Unseen-Easy | 53001; rails, 높이 4 cm, d=0.50 | 1.0 / 1.0 | 새 장애물 형상에서 보행 전이 |
| Unseen-Medium | 53012; gap 폭 설정 5–12 cm, d=0.50(폭 8.5 cm) | 0.65–1.35 / 0.55–1.15 | 지지면 단절 + 가변 마찰 |
| Unseen-Hard | 53003; 반복 cylinders, d=0.55 | 0.35–1.60 / 0.25–1.30 | 불규칙한 접촉 형상 + 넓은 마찰 범위 |

Hard cylinder의 보간 기준은 약 46개/tile, 높이 6.3 cm(±0.5 cm noise), 반지름 19.3 cm다. 학습 높이 상한보다 낮아도 형상·배치·접촉 변화가 달라 어려울 수 있다. 난이도 명칭은 순위를 보장하는 측정값이 아니다.

네 환경은 8 × 8 m 타일, 40행 × 20열 mesh를 쓰고 초기 행을 0–4로 제한했다. 생성기의 `curriculum=True`는 행 기반 배치 설정이며 평가 중 성과 기반 승격을 뜻하지 않는다. +x 방향으로 최소 304 m의 생성 지형을 확보하는 설계로, 작은 mesh 밖 평탄부로 빠지는 영향을 줄였다. 실제 개별 발 접촉 경로를 기록한 것은 아니다. 환경별 설정 원본은 [자료 표](#artifacts)에 연결했다.

### 결과

| 환경 | Episodes | Return 평균 ± std | 평균 steps | 960-step 생존 | 평균 +x 변위 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seen-Control | 100 | **59.264 ± 9.762** | 933.81 | **96%** | 53.051 m |
| Unseen-Easy | 100 | **57.186 ± 14.420** | 900.98 | **88%** | 51.671 m |
| Unseen-Medium | 100 | **60.527 ± 12.341** | 912.36 | **90%** | 54.309 m |
| Unseen-Hard | 100 | **19.225 ± 14.967** | 443.83 | **14%** | 17.228 m |

### 결과 해석

**1. 새 형상으로의 전이는 일부 조건에서 유지된다.** Easy와 Medium의 return은 Seen 대비 각각 −3.5%, +2.1%이며 전진 거리도 비슷하다. 이는 해당 checkpoint가 학습에 없던 rails·gap에서도 전진할 수 있다는 관측이다. 그러나 생존율은 각각 8%p, 6%p 낮다. 평균 점수가 비슷하다는 사실만으로 안정성까지 같다고 결론 내릴 수 없다.

**2. Hard의 핵심은 전진 감소와 episode 단축이다.** Seen 대비 return은 67.6%, 전진 거리는 67.5% 감소했고, 평균 지속 시간은 15.56초에서 7.40초로 줄었다. 86개 episode가 960 steps에 도달하지 못했다. 단순히 에너지 비용이 커진 결과라기보다 진행과 생존이 함께 약화된 양상이다.

| 평균 누적 reward 구성 | Seen-Control | Unseen-Hard | 해석 |
| --- | ---: | ---: | --- |
| Progress | 53.028 | 17.233 | 총 return 차이 40.039 중 약 35.794(89.4%)가 progress 감소와 대응 |
| Alive | 7.781 | 3.691 | episode 단축과 일치 |
| Energy penalty | −7.038 | −3.343 | 절댓값 감소는 짧아진 episode의 영향도 받음; 효율 개선 근거가 아님 |

**3. 가능한 기전과 관측을 구분한다.** 반복 cylinder의 곡면·기울기·배치와 마찰 변화가 발 지지와 자세 회복을 어렵게 했다는 설명은 타당한 *가설*이다. Hard의 동마찰 하한 0.25는 Curriculum 설정 하한 0.3보다 낮고, 팀 평가는 학습과 달리 정지·동마찰도 서로 다를 수 있다. 하지만 지형과 마찰이 동시에 바뀌어 어느 요인이 주원인인지 분리할 수 없다. 미끄러짐·충돌·자세 궤적을 측정하지 않았으므로 특정 실패 동작을 확인했다고 쓰지 않는다.

**4. Medium > Easy는 모순이 아니다.** gap과 rails는 단일 난이도 축의 두 단계가 아니다. 이 정책에는 gap 통과가 유리했을 가능성이 있지만, 다른 지형·마찰 샘플과 episode 분산도 영향을 준다. 단일 seed의 평균 차이만으로 난이도 역전이나 유의한 우위를 주장하지 않는다.

<a id="hypotheses"></a>
## 3. 가설 검증과 연구적 한계

| 가설 | 현재 근거 | 판정 / 필요한 추가 대조 |
| --- | --- | --- |
| H1: Terrain DR이 미지 지형에 도움 | Curriculum이 rails·gap에서 전진; flat PPO의 동일 평가 없음 | 전이 가능성은 관찰, **Terrain의 단독 효과는 미검증** |
| H2: Friction DR이 낮은 마찰에 도움 | 가변 마찰 조건의 결과는 있으나 fixed-friction 대조 없음 | **미검증**; geometry 고정 후 마찰만 변화, reset perturbation도 분리 |
| H3: Curriculum이 학습·일반화 개선 | 3-seed 학습 로그와 Curriculum seed 0 평가 | **미검증**; 같은 지형·마찰 최종 분포의 non-curriculum 대조 필요 |
| H4: History가 접촉 적응 개선 | 마지막 학습 return 40.193 vs 40.211 | 학습 평균의 추가 이득 불명확; **일반화는 미검증** |

- **평가 반복 범위:** training 3 seeds의 학습 로그와 달리 최신 평가는 training seed 0 × CLI seed 24 한 조합이다. 100개 병렬 episode는 100번의 독립 학습 또는 독립 mesh 반복이 아니다.
- **종료 규칙:** 원본의 world-frame 높이 <0.31 m 기준을 유지했다. 지면 대비 몸체 높이가 아니므로 지형 고저차가 종료 여부에도 영향을 줄 수 있다. 생존율은 이 규칙 아래의 지속률이다.
- **선정 근거:** 5 variant 전체의 공통 held-out 평가 자료가 없다. 따라서 “Curriculum이 가장 우수하다”거나 “seed 안정성이 가장 좋다”는 순위를 제시하지 않는다. 팀 평가에는 기존 제출 checkpoint를 고정 사용했다.
- **후속 검증:** 5 variant × 3 training seeds를 같은 평가 mesh들에서 평가하고, geometry-only / friction-only 조건을 분리해야 구성요소별 기여를 검증할 수 있다. 현재 Hard를 보고 조정한 모델은 별도의 새 held-out mesh에서 검증해야 한다.

## 4. 기존 Combined 평가 — 별도 보관

2026-10-02의 [원본 JSON](../results/curriculum_seed0_combined.json)은 새 팀 평가와 합산하지 않는다.

| 모델 | Suite / seed | Episodes | Return 평균 ± sample std |
| --- | --- | ---: | ---: |
| Curriculum seed 0, model_999.pt | Combined / 30000 | 100 | **29.118270 ± 12.040080** |

[overlay 평가 스크립트](../experiments/ant_generalization/play_one_episode.py)로 deterministic inference를 수행했다. 각 env의 첫 episode만 집계하며 online update는 없다. Noise/Blocks 높이 설정 8–12 cm, block 폭 30 cm, 마찰 0.2를 사용했다. 이 결과는 낮은 마찰과 지형 범위 변화 아래 얻은 점수이며, 대응 baseline 없이 개선량을 산출할 수 없다.

팀 Hard와는 geometry family·마찰·seed·mesh 크기·reset 설정·재질 적용 부위가 다르다. Combined 29.118과 Hard 19.225의 차이를 모델 퇴보나 단일 난이도 효과로 해석하지 않는다. 기존 mesh는 10행 × 20열이며 팀 평가와 같은 장거리 지형 통제를 보장하지 않는다.

<a id="artifacts"></a>
## 5. 원본 자료와 재현

`results/README.md`의 평가 조건·파일 안내는 이 절과 위 Combined 절로 통합했다.

| 팀 v2.1 환경 | Episode 원본 | 실행 env 설정 | Agent 설정 | 실행 로그 |
| --- | --- | --- | --- | --- |
| Seen-Control | [JSON](../results/team_eval_v2_1_20261005/seen_control.evaluation.json) | [YAML](../results/team_eval_v2_1_20261005/seen_control.evaluation.env.yaml) | [YAML](../results/team_eval_v2_1_20261005/seen_control.evaluation.agent.yaml) | [log](../results/team_eval_v2_1_20261005/Seen-Control.evaluation.log) |
| Unseen-Easy | [JSON](../results/team_eval_v2_1_20261005/unseen_easy.evaluation.json) | [YAML](../results/team_eval_v2_1_20261005/unseen_easy.evaluation.env.yaml) | [YAML](../results/team_eval_v2_1_20261005/unseen_easy.evaluation.agent.yaml) | [log](../results/team_eval_v2_1_20261005/Unseen-Easy.evaluation.log) |
| Unseen-Medium | [JSON](../results/team_eval_v2_1_20261005/unseen_medium.evaluation.json) | [YAML](../results/team_eval_v2_1_20261005/unseen_medium.evaluation.env.yaml) | [YAML](../results/team_eval_v2_1_20261005/unseen_medium.evaluation.agent.yaml) | [log](../results/team_eval_v2_1_20261005/Unseen-Medium.evaluation.log) |
| Unseen-Hard | [JSON](../results/team_eval_v2_1_20261005/unseen_hard.evaluation.json) | [YAML](../results/team_eval_v2_1_20261005/unseen_hard.evaluation.env.yaml) | [YAML](../results/team_eval_v2_1_20261005/unseen_hard.evaluation.agent.yaml) | [log](../results/team_eval_v2_1_20261005/Unseen-Hard.evaluation.log) |

[팀 취합 CSV](../results/team_eval_v2_1_20261005/result_template.csv) · [평가 checkpoint](../checkpoints/curriculum_seed0_model_999.pt) · [실행 환경](../ENVIRONMENT.md)

Checkpoint SHA-256: `38c4a31c087a02580c4499e76d377884ead4aa75317889d68808d58b395a0dc3`.

팀 평가 원본 위치는 Isaac Lab workspace의 `experiments/ant_team_eval_v2_1/runs/20261005_195743/`다. JSON·YAML·log 안의 절대 경로는 실행 당시 기록이다. **제출 overlay에는 팀 v2.1 wrapper·환경 배포본·팀 평가 스크립트가 포함되지 않는다.** 아래 팀 명령은 해당 코드가 설치된 workspace에서 실행하며, 기존 Combined용 스크립트와 구분한다.

```bash
# 팀 v2.1 코드가 설치된 Isaac Lab workspace에서 실행 (Seen-Control 예)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Curriculum-Team-v2_1-Seen-Control-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint Robotics_Simulation_8/checkpoints/curriculum_seed0_model_999.pt \
  --headless
```

나머지 task suffix는 `Unseen-Easy-v0`, `Unseen-Medium-v0`, `Unseen-Hard-v0`다. 기존 Combined 실행과 학습·그림 재생성 명령은 [README](../README.md#reproduce)에 있다.
