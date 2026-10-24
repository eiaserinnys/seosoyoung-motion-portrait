# 머리카락 저자유도 보정 최종 보고

상향 포즈의 넓고 평평한 정수리와 귀 뒤로 내려오던 삼각형 후두부를 교정했다. 각 점의 `correctedLift` 깊이를 `topZ`로 보간하여 볼록성과 y310 접속을 유지한다. 후두부 Y는 앞머리와 옆머리가 공유하는 높이에 유계 보충 변위만 더한다. 깊은 두개골의 raw pitch Y와 15도 Y collar를 제거했고 X/Z 접합과 중립 연속성은 보존했다.

## 최종 설정

`config/rig-config.json`은 수동 보정 설정의 정본이다. `pipeline/fit_dome.py`가 이 파일의 pitchArc, tipLag, rearPitch를 읽어 `source/viewer/dome-fit.json`과 `bundle/dome-data.mjs`에 반영한다. up-right 계수는 세 경로에서 모두 0이다.

- `up-left`: `rx=23.55969171144356, rz=27.220230095679536, y=0.0`
- `up-right`: `rx=0, rz=0, y=0`
- `down-left`: `rx=44.7706274656728, rz=4.537721491006616, y=0.0`
- `down-right`: `rx=30.23614894778265, rz=33.07390222556095, y=-4.1100229173979805`

## 승인한 증거 재사용

마지막 전체 H0 실행은 up-right 한 행의 이동 46px로 **exit 1**이었다. 최종 상태에 대한 새 H0나 WebGL 캡처는 실행하지 않았다. 사용자 승인에 따라 up-right의 후두부 보충만 0으로 고정하고, 해당 포즈는 이번 수정의 보충0 stage-a 실측과 타일을 재사용했다. 다른 8포즈는 latest-full-capture의 실측과 타일을 유지했다.

- 같은 런타임 3파일의 해시와 9포즈 실효 rx/rz/y가 재사용 원본과 완전히 일치한다.
- DOME의 rear 키만 원본 값으로 되돌려 직렬화한 해시도 두 실측의 기록과 일치한다. 다른 깊이·위도·지체 설정이 바뀌지 않았음을 확인했다.
- 각 시트의 up-right 타일 밖 픽셀 변화는 0이다. donor/final SHA256과 교체 사각형은 `hair-evidence-reuse.json`에 기록했다.
- 최종 판정은 **승인한 포즈별 증거 재사용에 따른 bounded pass**다. 보충0 대비 외곽 이동 최대는 **40px ≤ 45px**다. up-left40, up-right1, down-left19, down-right27px이며 나머지는0~1px다.

## 확인 결과

- 수식 교정 뒤 Node 전체 **140 passed**, Python 전체 **32 passed**.
- 최종 config/생성 데이터/up-right0 확인 및 관련 회귀는 Node **14 passed**.
- 독립 코드 감수와 설정·증거 재사용 증분 감수 통과.
- 얼굴/렌더러/리그/아틀라스/원화 보호19파일과 얼굴·피처 단언13본문은 main89fc408과 바이트 동일하다.
- 중립3포즈 픽셀 변화0과 정점/후두부/단면 해시 일치를 유지한다.
- 최종3시트를 직접 열어 상향 정수리의 네모난 지붕/좌우 뿔과 귀 뒤 긴 갈색 플랩이 사라진 것을 확인했다. 하단과 중립도 같은 프레임에서 비교했다.

원화/SAM 정합의 원시 미달 조건 22개는 `hair-r15-metrics.json`에 유지했다. 정확한 원화 정합을 통과했다고 표기하지 않는다. 전문 UI/UX 검수와 머지·발행은 담당 세션에 인계한다.

## 산출물 SHA256

```text
ea46ddeb87634f7f2b63ee9fa3870677cbfc758519548d1b74a4c4fec6b67ab4  bundle/index.html
2b7db3b58e73b9fba3dd4f607375c6f0852d6790a5989cbb9e63b0911fcc5da1  docs/captures/faceless_hair_9dir_comparison.png
9af9ff09ea27a9f8444d16273f16b0316ffbd6651aaee7dc867a65638394975d  docs/captures/hair_only_9dir_overlay.png
adb5a6d4b73915b90a60cbf0da91ed40e56aa0218457b9bb473e4f627d7c4288  docs/captures/viewer_9dir_grid.png
```

요청한 3시트는 `captures/viewer_9dir_grid.png`, `captures/faceless_hair_9dir_comparison.png`, `captures/hair_only_9dir_overlay.png`다. 원시 수치, 보충0 기준, 보정 계수와 증거 출처는 각각 `hair-r15-metrics.json`, `hair-stage-a-metrics.json`, `hair-low-dof-calibration.json`, `hair-evidence-reuse.json`에 보존했다.
