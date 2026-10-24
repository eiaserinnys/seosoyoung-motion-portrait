# 머리카락 줄기 연결·구면 정수리·대칭 보완

**이번 결과는 완료 기준 미달이다.** 단위 테스트와 저자유도 상한은 통과했지만 단일 H0는 exit1이었다. 줄기4곳과 up-right 상단 게이트가 남아 있으며, 재실행 상한에 따라 추가 H0나 캡처 뒤 제품 수정은 하지 않았다.

`6b9a6cf670736f0f3d443acac364be6979b15bf3`의 코드에서 H0를 **한 번** 실행해 9포즈와 캡처 3종을 생성했다. 타일 합성이나 이전 포즈의 실측 재사용은 없다. 캡처 뒤 제품 코드를 변경하지 않았으며 `hair-r15-metrics.json`의 productSha256으로 최종 코드와 대조한다.

## 원인과 보완

방울은 중력 프레임에 남지만 뒤통수·뒷머리는 피치에 따라 들렸고, 옆머리는 방울 중심 y544에서 Y만 수렴했다. 세 계층 모두 y405부터 방울 윗선 y506까지 구면 이징으로 중력 프레임 X/Y에 수렴한다. 먼 쪽 가닥은 원화 방울의 중앙 60% 폭까지 확장한다. 피치·yaw 이징을 함께 곱해 중립 3포즈와 정면 통과의 연속성을 유지한다. 열별 stem 측정은 각 열의 방울 상단에서 위로 최대80px를 스캔한 최악값이다.

정수리는 각 점의 볼록한 base 깊이에서 정면 topZ=-90, 대각선 diagonalTopZ=-25로 보간한다. yaw 가중치는 sphereEase(abs(correctedYaw)/45)다. `tipLag` 시작점을 y230으로 둔 이유는 앞머리 갈래 틈과 hairLift의 lowerMix가 y230부터 시작하기 때문이다.

## 설정과 보호 계약

설정 정본은 `source/viewer/hair-pitch.json`이다. fit_dome.py가 dome-fit.json과 dome-data.mjs를 생성한다. config 디렉터리는 제거했다. tipLag는 up=.35, down=.35, up-diag=.6, down-diag=.85이며 기본값은 .65다. 양쪽 up rear는0, 양쪽 down rear는 rx=44.7706274656728, rz=4.537721491006616, y=0을 공유한다.

보호19파일은 main89fc408과 바이트 동일하다. 혼합 테스트의 얼굴/피처 단언은 유지했다. 중립 left/front/right 픽셀 및 정점/후두부/단면 비교는 H0의 개별 지표에 기록했다. UI 마크업·공통 레이아웃·원화·아틀라스는 변경하지 않았다.

## 단일 H0 결과

판정: **fail**, pass=false. 보충0 대비 외곽 이동 최대 **30px** / 상한45px. 중립 픽셀 변화와 정점 해시, 상향 정수리, 피치6포즈 양쪽 줄기 틈은 필수 게이트다. 다른 참조/SAM 잔차는 승인한 저자유도 bounded pass 정책에 따라 원시값을 남긴다. 지정 topZ=-90 때문에 down의 기존 상단≤4px 기준에 남는 잔차도 숨기지 않는다.

| 포즈 | 상단 Y | 왼쪽 열별 줄기 틈 | 오른쪽 열별 줄기 틈 | 외곽 이동 최대 |
| :--- | ---: | ---: | ---: | ---: |
| up-left | 60 | 81 | 1 | 0 |
| up | 97 | 4 | 5 | 1 |
| up-right | 61 | 1 | 14 | 0 |
| left | 10 | 81 | 1 | 0 |
| front | 10 | 4 | 5 | 1 |
| right | 10 | 1 | 81 | 0 |
| down-left | 41 | 16 | 1 | 30 |
| down | 7 | 5 | 1 | 1 |
| down-right | 39 | 1 | 81 | 18 |

원시 참조 게이트 미달 24개:

- up-left / hairline median: `12.5`
- up-left / forehead exposure: `0.10940868493994449`
- up-left / posterior exterior: `44`
- up-left / left stem gap: `81`
- up / upper exterior: `86`
- up / hairline median: `-20.0`
- up / forehead exposure: `0.11308784093138868`
- up-right / crown top: `61`
- up-right / hairline median: `-39.0`
- up-right / posterior exterior: `72`
- up-right / right stem gap: `14`
- down-left / crown top: `41`
- down-left / forehead exposure: `0.11114023591087818`
- down-left / posterior exterior: `107`
- down-left / left stem gap: `16`
- down / crown top: `7`
- down / upper exterior: `43`
- down / forehead exposure: `-0.1463126843657817`
- down-right / crown top: `39`
- down-right / hairline median: `-13.5`
- down-right / forehead exposure: `-0.2767023850674041`
- down-right / posterior exterior: `152`
- down-right / near cheek background gap: `{"sampledRows": 61, "maximumGap": 1}`
- down-right / right stem gap: `81`

## 검증과 산출물

- 전체 Node144 passed, Python34 passed. 이후 yaw 연속성/경계 게이트 변경은 영향 Node16 passed, 측정기11 passed로 확인했다.
- 독립 코드 감수 통과. far 폭 확장의 yaw0 활성화 점프를 보완하고 연속성 회귀를 추가했다.
- metrics는 요약 지표만 저장한다. 행별 상세는 git에서 제외되는 debug/hair_h0/current/hair-r15-metrics-raw.json에 저장한다. 이전 stage-a-metrics와 evidence-reuse 문서는 삭제했다.
- 캡처는 captures/viewer_9dir_grid.png, captures/faceless_hair_9dir_comparison.png, captures/hair_only_9dir_overlay.png다. 전문 UI/UX 재확인과 머지는 담당 세션에 인계한다.

3시트를 직접 열어 전체9포즈와 머리카락 단독 참조를 비교했다. 정면 up 상단은 둥글지만 up-right의 참조 대비 상단 차이와 먼 쪽 방울 위 빈 구간이 남아 있다. down 대각선의 후두부 외곽도 참조와 차이가 크다. 실화면 완료나 UI/UX 재검수 통과로 보고하지 않는다.

## SHA256

```text
ea46ddeb87634f7f2b63ee9fa3870677cbfc758519548d1b74a4c4fec6b67ab4  bundle/index.html
d7bee6858003dbe5999bb78bdc9fac8dcf12ba5516088b375a35fb584c218d4a  docs/captures/faceless_hair_9dir_comparison.png
b44324a091e50ef555f1e079ea6e27562d951a986506fbf4edbef3caec71951c  docs/captures/hair_only_9dir_overlay.png
e450bc0d9d6a3f624a64d5dba4eca6476f5c970b1f89968d4c0a5bd3308ceccf  docs/captures/viewer_9dir_grid.png
```
