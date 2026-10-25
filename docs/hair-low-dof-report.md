# 구슬 기준 상부·하부 머리카락 볼륨 피팅

6포즈 평균 실제 래스터 RMS: 이전 main 36.878px → 20.483px.
트윈테일 스팬 RMS: prior 54.848px → 40.021px.
상부 차트 minDetJ 0.274784, 하부 실제 삼각형 minDetJ 0.255891. 역전 0개.
남은 절대 원화 게이트 실패: 14개. 상세는 `hair-r15-metrics.json`의 rawResidualFailures에 보존했다.

| 포즈 | 이전 main RMS | 보정 RMS | 하부 prior RMS | 하부 보정 RMS |
| :--- | ---: | ---: | ---: | ---: |
| up-left | 35.693 | 22.283 | 49.430 | 35.218 |
| up | 36.445 | 18.493 | 54.678 | 42.956 |
| up-right | 37.025 | 17.299 | 69.734 | 51.090 |
| down-left | 42.703 | 34.882 | 48.633 | 35.884 |
| down | 18.748 | 14.188 | 34.648 | 24.330 |
| down-right | 50.652 | 15.751 | 71.967 | 50.648 |

## 계산과 불변 조건

- 상부: crown | face만 사용한 외곽과 헤어라인, 볼 옆 경계. 12×16 차트, restY515..545 감쇠. 변형 후 외곽을 재측정하는 3회 피팅.
- 하부: 좌우 구슬 원점의 8노드 중심 이동·폭 스케일. 구슬·매듭을 포함하는 첫 3노드는 항등 변환. 나머지 갈래를 별도로 피팅.
- 무접힘: 상부는 위반 셀의 야코비안을 국소 투영한 뒤 독립 JS 게이트로 검증. 하부는 실제 네 메시 전체 삼각형을 45개 기준·중간·측면 포즈에서 검증.
- 중립 3포즈: 정점 해시 일치와 변경 픽셀 0. 구슬 기준점은 모든 검사 포즈에서 정확히 일치. stemGap ≤17px.
- 동결: hair-shell, hair-surface, dome-data, rig JSON, texture 및 얼굴 소스는 기준 main과 바이트 동일.
- 뷰어 컴포넌트·글자·여백·조작은 변경하지 않았다. 같은 전체 초상 프레임에서 원화와 렌더를 비교했다.

## 남은 절대 오차의 근거

- y520의 up-right 외곽은 crown이 아닌 고정 ties의 [263,754]이며 원화는 [203,721]이다. 왼쪽60px 차이는 원화의 눈 등록 이동[-56.65,+28.46]과 구슬 dBodyFrame 고정 프레임이 만나는 행에서 측정된다. up-left의 같은 행도 ties가 외곽을 소유하며 오른쪽29px 차이가 남는다.
- bare-face의 alpha 삼각형을 머리카락 없이 스캔했다. down x525..549는 원화 상단265..269에 비해 bare-face 상단305..306, down-right x669는 원화241에 비해 bare-face290이다. 머리카락만 움직여 이 픽셀에 피부를 드러낼 수 없다.
- 정수리·다른 상부 외곽의 피팅 잔차도 남아 있다. 전체 14개 값과 열별 하한은 vdd-research-evaluation.json의 remainingEvidence에 보존했다.
- 담당 승인에 따라 원화/게이트를 이동·제외하지 않고 평균 20.48px 후보를 채택했다. 19.54px 후보에서 가려졌던 down-left의 330행 피부가 보이고 헤어라인 중앙 오차가 2px로 측정된다.

## 검증

Node 189 passed / 0 failed. Python 103 passed, 15 subtests passed.
실화면: captures/faceless_hair_9dir_comparison.png, hair_only_9dir_overlay.png, r15_before_after_9dir.png, vdd_unobserved_angles_sheet.png.

## 재현

무거운 명령은 heavy runner로 감싼다. 워크트리 .venv에 requirements.txt와 pytest-timeout을 준비한다.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python pipeline/fit_hair_solve.py --capture-prior
OPENBLAS_NUM_THREADS=1 .venv/bin/python pipeline/fit_hair_solve.py
.venv/bin/python pipeline/fit_hair_solve.py --capture-base 4dba92d65c74d1ac13dfa31ac791bf2f568c2140
.venv/bin/python pipeline/probe_hair_only.py
node pipeline/vdd_benchmark.mjs
node --test --test-concurrency=2 --test-timeout=60000 pipeline/test_*.mjs tests/*.test.mjs
.venv/bin/python -m pytest --timeout=60 -q pipeline tests
.venv/bin/python pipeline/update_vdd_evidence.py
```
