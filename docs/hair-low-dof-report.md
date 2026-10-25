# 구슬 기준 상부·하부 머리카락 볼륨 피팅

6포즈 평균 실제 래스터 RMS: 이전 main 37.881px → 21.689px.
트윈테일 스팬 RMS: prior 54.838px → 46.984px.
상부 차트 minDetJ 0.285334, 하부 실제 삼각형 minDetJ 0.592650. 역전 0개.
남은 절대 원화 게이트 실패: 14개. 상세는 `hair-r15-metrics.json`의 rawResidualFailures에 보존했다.

| 포즈 | 이전 main RMS | 보정 RMS | 하부 prior RMS | 하부 보정 RMS |
| :--- | ---: | ---: | ---: | ---: |
| up-left | 35.366 | 23.893 | 49.430 | 40.307 |
| up | 36.400 | 22.507 | 54.678 | 48.112 |
| up-right | 38.282 | 22.791 | 69.734 | 56.499 |
| down-left | 47.787 | 25.889 | 48.574 | 43.004 |
| down | 18.799 | 15.302 | 34.648 | 30.321 |
| down-right | 50.652 | 19.752 | 71.967 | 63.661 |

## 계산과 불변 조건

- 상부: crown | face 외곽과 헤어라인, 볼 옆 경계. 12×16 차트, restY515..545 감쇠. 변형 후 외곽을 재측정하는 3회 피팅. 헤어라인은 25열 중앙값·수직 법선·가중치6, 외곽 법선 기울기는 ±1.2로 제한한다.
- 내부 결: 수평2차차분28, scale0.15, conformal0.35로 인접 열의 전단을 억제한다. 머리핀의 pitch 잔차는 [598,290]에서 구한 동일 평행이동만 적용하며 기존 측면 투영과 canopy 모양은 보존한다.
- 하부: 좌우 구슬 원점의 8노드 중심 이동·폭 스케일. 구슬·매듭을 포함하는 첫 3노드는 항등 변환. 2차차분120·1차차분8로 부드럽게 피팅하고 폭 배율을 0.75..1.35로 제한한다.
- 결합부: prior의 실제 alpha 경계 중 구슬 위를 지지하는 차트 노드를 정확히 고정한다. 주변 정점에 대한 연성 관측만으로 남던 0.12px 이동이 얇은 연결 픽셀을 잃게 하는 문제를 제거했다.
- 무접힘: 상부는 고정 노드를 보존하며 위반 셀의 야코비안을 국소 투영한 뒤 독립 JS 게이트로 검증. 하부는 실제 네 메시 전체 삼각형을 45개 기준·중간·측면 포즈에서 검증.
- 중립 3포즈: 정점 해시 일치와 변경 픽셀 0. 구슬 기준점은 모든 검사 포즈에서 정확히 일치. stemGap ≤17px.
- 동결: hair-shell, hair-surface, dome-data, rig JSON, texture 및 얼굴 소스는 통합 main 17032ea와 바이트 동일.
- 뷰어 컴포넌트·글자·여백·조작은 변경하지 않았다. 같은 전체 초상 프레임에서 원화와 렌더를 비교했다.

## 남은 절대 오차의 근거

- y520의 구슬 결합부는 눈 등록 이동이 적용된 원화와 dBodyFrame 고정 좌표계가 만나는 행이다. 최신 포즈별 render/ref 스팬·crown/face/ties 소유 범위·등록 이동은 remainingEvidence.tieRow520에 다시 측정했다.
- 이전 얼굴 정합에서 측정했던 bare-face 경계 하한은 폐기했다. Track F 통합 후 머리카락을 제거한 최신 face alpha 삼각형을 다시 스캔한 열별 하한만 remainingEvidence.bareFace에 기록한다.
- 정수리·다른 상부 외곽의 피팅 잔차도 남아 있다. 전체 14개 값과 열별 하한은 vdd-research-evaluation.json의 remainingEvidence에 보존했다.
- 보완 요청에 따라 원화/게이트를 이동·제외하지 않고 내부 결의 부드러움과 평균 21.69px 볼륨 개선을 함께 검증했다.

## 검증

Node 193 passed / 0 failed. Python 109 passed, 21 subtests passed.
실화면: captures/faceless_hair_9dir_comparison.png, hair_only_9dir_overlay.png, r15_before_after_9dir.png, vdd_unobserved_angles_sheet.png.

## 재현

무거운 명령은 heavy runner로 감싼다. 워크트리 .venv에 requirements.txt와 pytest-timeout을 준비한다.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python pipeline/fit_hair_solve.py --capture-prior
OPENBLAS_NUM_THREADS=1 .venv/bin/python pipeline/fit_hair_solve.py
.venv/bin/python pipeline/fit_hair_solve.py --capture-base 17032ea
.venv/bin/python pipeline/probe_hair_only.py
node pipeline/vdd_benchmark.mjs
node --test --test-concurrency=2 --test-timeout=60000 pipeline/test_*.mjs tests/*.test.mjs
.venv/bin/python -m pytest --timeout=60 -q pipeline tests
.venv/bin/python pipeline/update_vdd_evidence.py
```
