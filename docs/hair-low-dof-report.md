# 우하 얼굴 정본화와 머리카락 합동 피팅

비악화 게이트 판정: **monotone prior pass**. 3D 사전은 `0b75a0211691965ca234ad9114073ba934568394`와 바이트가 같다. 절대 원화 목표 전부 통과: False.

2D 차트: True; 잔차 예산: True; 중립 머리카락 정점 불변: True.

## 방향별 실측

| 포즈 | 상단Y | 앞머리 오차 | 이마 면적비 오차 | 볼 틈 | 잔차 RMS / 최대 | 차트 minDet |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| up-left | 54 | -4.0 | 0.1031554286550993 | 92 | 3.61 / 13.75 | 0.9044 |
| up | 98 | -8.5 | -0.0927300652758224 | 해당 없음 | 5.86 / 18.30 | 0.7560 |
| up-right | 53 | -33.0 | -0.025736224791156892 | 93 | 4.96 / 16.72 | 0.6304 |
| left | 10 | -16.0 | 0.04118259754827336 | 0 | 0 / 0 | 1 |
| front | 10 | -4.0 | 0.12455197132616491 | 해당 없음 | 0 / 0 | 1 |
| right | 10 | -30.5 | -0.022742100848168523 | 0 | 0 / 0 | 1 |
| down-left | 12 | -7.0 | 0.09154145798553137 | 3 | 4.35 / 11.52 | 0.9450 |
| down | 3 | -6.5 | -0.06404745529573586 | 해당 없음 | 4.66 / 12.94 | 0.9725 |
| down-right | 9 | -9.0 | -0.29743746186699205 | 4 | 5.32 / 16.25 | 0.9128 |

## 동일 사전 대비 실제 렌더 오차

actual raster RMS in pixels: head L/R spans at y80..400 step20, topY, and hairline columns step12; fixed reference/prior-visible domain

| 포즈 | 사전 RMS | 보정 RMS | 판정 |
| :--- | ---: | ---: | :--- |
| up-left | 38.260 | 35.693 | 통과 |
| up | 42.165 | 36.445 | 통과 |
| up-right | 41.277 | 37.025 | 통과 |
| left | 40.251 | 40.251 | 통과 |
| front | 21.991 | 21.991 | 통과 |
| right | 31.507 | 31.507 | 통과 |
| down-left | 52.651 | 42.703 | 통과 |
| down | 22.784 | 18.748 | 통과 |
| down-right | 60.582 | 50.652 | 통과 |

## 남은 원화 목표 오차

[
  {
    "pose": "up-left",
    "gate": "posterior exterior",
    "value": 29
  },
  {
    "pose": "up",
    "gate": "upper exterior",
    "value": 74
  },
  {
    "pose": "up-right",
    "gate": "hairline median",
    "value": -33.0
  },
  {
    "pose": "up-right",
    "gate": "posterior exterior",
    "value": 60
  },
  {
    "pose": "down-left",
    "gate": "crown top",
    "value": 12
  },
  {
    "pose": "down-left",
    "gate": "posterior exterior",
    "value": 64
  },
  {
    "pose": "down-left",
    "gate": "near cheek background gap",
    "value": {
      "sampledRows": 61,
      "maximumGap": 3
    }
  },
  {
    "pose": "down",
    "gate": "upper exterior",
    "value": 34
  },
  {
    "pose": "down-right",
    "gate": "crown top",
    "value": 9
  },
  {
    "pose": "down-right",
    "gate": "forehead exposure",
    "value": -0.29743746186699205
  },
  {
    "pose": "down-right",
    "gate": "posterior exterior",
    "value": 111
  },
  {
    "pose": "down-right",
    "gate": "near cheek background gap",
    "value": {
      "sampledRows": 61,
      "maximumGap": 4
    }
  }
]

## 재생성

얼굴: fit_features.py → fit_outline.py → optimize_3d_2d.py → solve_to_bundle.py. 머리카락: capture_hair_solve_prior.py → fit_hair_solve.py → accept_hair_solve.py → probe_hair_only.py. 캡처 명령은 heavy runner에서 실행한다.

기본 격자 정규화: {'ridge': 0.5, 'xSecondDifference': 2, 'ySecondDifference': 50}. 실제 포즈별 계수는 hair-solve.json의 regularization에 있다. 중립 3포즈와 restY435 이상은 보정 0이다. 3D 사전·파츠·텍스처·메시 토폴로지는 변경하지 않았다.

현재 수용 기준은 실제 렌더 오차의 개선 또는 동등, 중립 불변, 2D 차트 무접힘과 잔차 예산이다. 과거 cap과 절대 원화 오차는 진단값으로 보존하며 위 남은 원화 오차가 모두 해소됐다는 뜻은 아니다.

## VDD 연구 적용과 일반화

실제 6포즈 평균 래스터 RMS: main 37.281200px → 36.877796px. 6포즈 모두 같은 r15 prior보다 개선됐다.

| 평가 | main | 개선 |
| :--- | ---: | ---: |
| 관측 normal RMS (px) | 38.146653 | 37.667946 |
| LOPO normal RMS (px) | 42.963929 | 42.320561 |
| 정면 각속도 점프 (px/deg, ε=0.0001) | 0.147000902 | 0.000000342 |
| 중간 각도 평균 CR 에너지 | 0.000547662 | 0.000867824 |
| 중간 각도 평균 대칭 RMS (px) | 0.878552384 | 0.827558262 |

기준·중간·연속 스윕의 머리카락 차트 minDetJ 최솟값은 0.630384이다. 중간 각도 CR 에너지는 증가했으며, 채택 기준은 래스터 평균 개선과 변위·무접힘 예산이다. 얼굴 데이터와 원래 3D prior는 바이트가 같다.

얼굴 8포즈의 Pin/LOPO는 `vdd-research-evaluation.json`의 crossValidation.face에 전후를 각각 기록했다. 얼굴은 변경하지 않아 값이 동일하다. 머리카락 LOPO는 보류 포즈의 데이터 항을 빼고 거울·이웃 시점 결합으로 예측한다. Pin/LOPO 관측 RMS와 실제 래스터 RMS는 측정 영역과 정의가 다르다.

### 적용과 보류

- C1 trigonometric yaw weights through zero; cardinal nine keys unchanged; smoothstep fade from 60 to 88 degrees isolates profile parts
- Silhouette confidence: 1.25 at y<=220, 0.55 at 260..360, otherwise 0.85; divide by 1+0.5*slope^2
- Area-weighted cell-center Cauchy-Riemann energy on (160*s,restY) pixels (lambda=0.5); mirror coupling between diagonal charts (lambda=0.2); weak adjacent-yaw chart prior (lambda=0.05) also predicts neutral LOPO
- Separate pin, held-out normal-observation prediction, raster error, sampled Jacobians, neutral-yaw derivatives and actual WebGL sweep

- 보류: POSTECH full 3D mesh reconstruction and ray deformation pipeline — Would replace the approved authored 3D prior, topology and texture workflow; residual chart improvements meet the present scope
- 보류: Inverse Rig neural MLP surrogate — The existing linear normal-equation solve is directly differentiable and small; training a surrogate adds approximation error and dependencies
- 보류: Fixed contour vertex IDs at +/-90 degrees — Visible silhouette identities change with occlusion and dedicated profile parts; preserve the existing measured profile path
- 보류: Full nonlinear ARAP — Discrete CR penalizes shear/anisotropic scale with a single linear solve; it permits uniform scale and rotation, so it is documented as conformal rather than equated to ARAP

### 근거 문헌

- [View-Dependent Deformation for 2.5-D Cartoon Models (2022)](https://doi.org/10.1109/MCG.2022.3174202): Reuse the existing trigonometric cardinal basis through yaw=0; this basis is a project choice, not a reproduction of the paper
- [3D Character Reconstruction from Hand-drawn Model Sheets (2026)](https://doi.org/10.1111/cgf.70323): Keep authored 3D geometry and constrain the residual chart with discrete CR energy. This linear conformal regularizer is a project adaptation, not the full mesh/Jacobian reconstruction pipeline
- [Inverse Rig Optimization from Line Drawings (2026)](https://arxiv.org/html/2609.00732v1): Retain the existing normal-distance loss, attenuate lower/steep silhouette samples as a visibility confidence heuristic; y/slope weights are project-specific, not learned occlusion or surface-normal estimation
- [View-Dependent Deformation Fields for 2D Editing of 3D Models (2025)](https://arxiv.org/html/2504.05544v1): Smooth neutral-yaw interpolation and independent 60..88 degree support fade; keep the existing additive residual rather than introducing compositional 3D deformation layers

### 검증 범위와 한계

- Twelve unobserved angle combinations have no independent reference drawings; their metrics measure orientation, conformal distortion and mirror consistency, not held-out image accuracy
- LOPO evaluates normal-distance observations on a fixed prior-visible domain; it is separate from the actual raster acceptance metric
- The unchanged shared 3D prior and face fit were calibrated from all original drawings. LOPO here withholds residual-chart observations, not every upstream drawing-derived asset
- C1 is guaranteed through yaw=0; the existing clamp at +/-45 remains. Support fading is C1 at 60 and 88
- Conformal energy at intermediate views can increase while raster error and symmetry improve; report measured before/after values without using this diagnostic as a fitted success claim

실화면: `captures/vdd_unobserved_angles_sheet.png`에서 main/after 같은 각도와 크기를 비교했다. 중간 대각선에서 새 틈·직사각형 돌출이 보이지 않았고 수평 4컷은 전후가 같다. 기존 뷰어 UI·컴포넌트·CSS·글자·간격은 변경하지 않았다. 캡처는 초상 전체를 같은 720×720 영역으로 잘라 같은 비율로 축소했다. 전문 UI/UX 검수는 담당 세션에 인계한다.

### 재현 순서

무거운 명령은 heavy runner 안에서 하나씩 실행한다. Python 검증은 pytest-timeout이 설치된 인터프리터를 사용한다.

```sh
python3 pipeline/capture_vdd_baselines.py
OPENBLAS_NUM_THREADS=1 python3 pipeline/fit_hair_solve.py
python3 pipeline/accept_hair_solve.py
node pipeline/vdd_benchmark.mjs
python3 pipeline/capture_vdd.py
python3 pipeline/probe_hair_only.py
python3 pipeline/update_vdd_evidence.py
```
