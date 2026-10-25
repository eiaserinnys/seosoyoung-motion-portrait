# 우하 얼굴 정본화와 머리카락 합동 피팅

비악화 게이트 판정: **monotone prior pass**. 3D 사전은 `0b75a0211691965ca234ad9114073ba934568394`와 바이트가 같다. 절대 원화 목표 전부 통과: False.

2D 차트: True; 잔차 예산: True; 중립 머리카락 정점 불변: True.

## 방향별 실측

| 포즈 | 상단Y | 앞머리 오차 | 이마 면적비 오차 | 볼 틈 | 잔차 RMS / 최대 | 차트 minDet |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| up-left | 54 | -4.0 | 0.1035214876638113 | 92 | 3.64 / 13.75 | 0.9030 |
| up | 98 | -4.0 | -0.04890506288705865 | 해당 없음 | 4.18 / 12.50 | 0.7882 |
| up-right | 52 | -31.0 | -0.02843641886760606 | 93 | 5.07 / 16.96 | 0.6208 |
| left | 10 | -16.0 | 0.04118259754827336 | 0 | 0 / 0 | 1 |
| front | 10 | -4.0 | 0.12455197132616491 | 해당 없음 | 0 / 0 | 1 |
| right | 10 | -30.5 | -0.022742100848168523 | 0 | 0 / 0 | 1 |
| down-left | 13 | -5.0 | 0.08708959376739012 | 3 | 4.93 / 12.82 | 0.9444 |
| down | 2 | 1.5 | -0.0599398883641048 | 해당 없음 | 4.96 / 13.79 | 0.9641 |
| down-right | 10 | -10.5 | -0.2784215471343564 | 4 | 3.33 / 9.83 | 0.9482 |

## 동일 사전 대비 실제 렌더 오차

actual raster RMS in pixels: head L/R spans at y80..400 step20, topY, and hairline columns step12; fixed reference/prior-visible domain

| 포즈 | 사전 RMS | 보정 RMS | 판정 |
| :--- | ---: | ---: | :--- |
| up-left | 38.260 | 35.661 | 통과 |
| up | 42.165 | 37.744 | 통과 |
| up-right | 41.277 | 35.322 | 통과 |
| left | 40.251 | 40.251 | 통과 |
| front | 21.991 | 21.991 | 통과 |
| right | 31.507 | 31.507 | 통과 |
| down-left | 52.651 | 41.796 | 통과 |
| down | 22.784 | 18.686 | 통과 |
| down-right | 60.582 | 54.478 | 통과 |

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
    "value": 78
  },
  {
    "pose": "up-right",
    "gate": "hairline median",
    "value": -31.0
  },
  {
    "pose": "up-right",
    "gate": "posterior exterior",
    "value": 60
  },
  {
    "pose": "down-left",
    "gate": "crown top",
    "value": 13
  },
  {
    "pose": "down-left",
    "gate": "posterior exterior",
    "value": 63
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
    "value": 10
  },
  {
    "pose": "down-right",
    "gate": "forehead exposure",
    "value": -0.2784215471343564
  },
  {
    "pose": "down-right",
    "gate": "posterior exterior",
    "value": 121
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
