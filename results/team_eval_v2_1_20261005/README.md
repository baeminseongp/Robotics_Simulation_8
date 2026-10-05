# 팀 공통 평가 v2.1 — 2026-10-05

Run: `20261005_195743` (Asia/Seoul). 기존 Curriculum seed 0 checkpoint를 CLI seed 24로 평가했다. 네 환경 각각 첫 episode 100개, 총 400개를 수집했다. 표준편차는 sample std(ddof=1)다.

| 환경 | Episode return 평균 ± 표준편차 | Steps 평균 | 960-step 생존율 | +x 변위 평균 (m) |
| --- | ---: | ---: | ---: | ---: |
| seen_control | 59.264 ± 9.762 | 933.81 | 96% | 53.051 |
| unseen_easy | 57.186 ± 14.420 | 900.98 | 88% | 51.671 |
| unseen_medium | 60.527 ± 12.341 | 912.36 | 90% | 54.309 |
| unseen_hard | 19.225 ± 14.967 | 443.83 | 14% | 17.228 |

Easy·Medium에서는 Seen과 비슷한 평균 return을 보였고, Hard에서는 return과 생존율이 크게 낮아졌다. 완료 100/100은 생존율이 아니다. 조기 종료도 완료 episode로 집계한다. 단일 평가 seed의 결과이며, 지형·마찰이 함께 바뀌므로 성능 차이의 원인을 하나로 단정할 수 없다.

## 평가 조건

- Checkpoint: [`../../checkpoints/curriculum_seed0_model_999.pt`](../../checkpoints/curriculum_seed0_model_999.pt)
- SHA-256: `38c4a31c087a02580c4499e76d377884ead4aa75317889d68808d58b395a0dc3`
- Deterministic inference, 관측 60 / 행동 8, 원본 7개 reward 가중치.
- 최대 960 policy steps = 16초. 각 parallel environment의 첫 episode만 집계.
- Task ID와 terrain seed, episode별 return·steps·+x 변위, reward 항목별 통계는 각 JSON에 기록.
- 기존 `combined` 평가는 suite와 seed가 다르므로 이번 팀 평가와 구분한다.

## 포함 자료

- [`result_template.csv`](result_template.csv): 네 환경 취합표 (`member=curriculum_seed0`).
- `*.evaluation.json`: 총 400 episode의 원시 수치와 평가 metadata.
- `*.evaluation.env.yaml`, `*.evaluation.agent.yaml`: 실행 당시 환경·정책 설정.
- `*.evaluation.log`: checkpoint 로드와 `Completed first episodes: 100/100` 확인용 원본 로그.

원본 위치: Isaac Lab workspace의 `experiments/ant_team_eval_v2_1/runs/20261005_195743/`. JSON·YAML·로그의 로컬 경로는 실행 당시 기록을 보존했다. 이 디렉터리는 결과 기록이며, 팀 v2.1 환경 코드와 평가 스크립트는 포함하지 않는다.
