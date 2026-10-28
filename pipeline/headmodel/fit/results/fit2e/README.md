# 2e 최종 정합: K = 9

2026-10-28 담당 판정으로 K=9를 최종 확정했다. 원본 결과는 커밋 `06f7b2fca472b806d4ed83576241492d516ec18b`에 있다.

## 최종 구성

- 공유 형상 V_B: B130-n3, `pipeline/headmodel/fit/results/fit2f/fit-state-n3.npz`.
- 보정장: 정규화 가우시안 9노드, σ=18°, pitch 거리 가중 1.4. 자기 가중은 실측만 기록한다.
- 최종 파일: `pipeline/headmodel/fit/results/fit2e/final-state.npz`, `pipeline/headmodel/fit/results/fit2e/residuals.json`. 각각 `k9-state.npz`, `k9-residuals.json`의 원문 복사다.
- 보정 포즈: `pipeline/headmodel/fit/results/fit2e/poses-refined.json` (`refine_rotation=true`).
- 제외 10장: 103, 301, 303, 310, 313, 316, 319, 322, 325, 328. 모두 보정 후 hair 중앙값 >8px이며, 훈련·검증·잡음 바닥에서 제외했다. 303 노드는 보정된 각도에 유지했다.
- 훈련 92장, 검증 28장. λ_k=1000s_k, μ_k=.01s_k, ρ=10000이며 ν는 기존 정의를 유지했다.

| 노드 프레임 | s_k |
| --- | ---: |
| 3 | 2.680325 |
| 42 | 2.574641 |
| 192 | 1.541517 |
| 206 | 1.206042 |
| 219 | .938846 |
| 240 | 1.133854 |
| 264 | .937230 |
| 270 | .490956 |
| 303 | 1.791021 |

## 채택 판정과 기록할 오차

담당이 keys-k9, rotation-k9, jaw-keys-k9를 직접 확인했다. 9방향 턱선은 원화와 겹치며 제외된 303도 겹친다. 검증 피부 중앙값 .573px는 측정 잡음 값 1.00px보다 낮고 삽입 조건 2가 거짓이므로, 노드 삽입과 σ 보정 없이 확정했다.

곡률 최대는 노드 264의 .068105, 270의 .076358로 기준 .05를 넘는다. 회전 시트의 pitch −25°, yaw 0~45°(5° 간격) 10칸도 초과하며 최대 .068209다. 위치는 턱밑 국소점으로 담당이 받아들였다. 원본 수치 검사 결과(`passed=false`)는 그대로 보존한다.

프레임 최대의 P95는 피부 훈련/검증 11.48/13.28px, hair 29.57/29.43px다. 피부는 턱밑과 목, hair는 귀 뒤 후두부 하단과 목이 만나는 경계의 오차로 기록하고 받아들였다. 상세 좌표·통계·검수는 `pipeline/headmodel/fit/results/fit2e/residuals.json`과 `pipeline/headmodel/fit/results/fit2e/validation-k9.json`에 있다.

귀 배치와 리그 연결은 별도 설계 대상이다.
