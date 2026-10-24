# 머리카락 구면 보정 최종 검증

최종 제품 코드 `0b6018d9e07191bdccaf0e5cca3ba4f66d066ef7`에서 H0 캡처를 한 번 실행했다. 이후 사용자가 승인한 down 상단≤13px와 Fable의 줄기 집계 정의로 같은 프레임을 재분석해 **bounded pass / exit0**를 확인했다. 최초 측정은 이전 게이트로 exit1이었으며, 추가 캡처 없이 측정 기준만 정정했다. 캡처 뒤 제품 코드 변경은 없고, productSha256과 measurementReassessment.probeSha256이 최종 코드와 측정기를 실측에 연결한다. 캡처3종은 이 단일 실행에서 생성했으며 이전 타일 합성은 없다.

## 하향 대각선 복원

pitchHairDepth의 yawWeight에 sphereEase(max(0,correctedPitch)/30)를 곱했다. 양의 피치에서만 diagonalTopZ=-25를 사용하고 음의 피치에서는 topZ=-90을 유지한다. 추가 스칼라는 없다. pitch0에서 값과 미분이 연속이며, 정면과 상향30도 키포즈는 유지한다. 이전 하향 대각선 상단41/39px 회귀를 복원했다.

## 필수 통과 게이트

- 9포즈가 모두 있고 중립 left/front/right 픽셀 변화0, 전체 정점·후두부·단면 해시 일치.
- up 상단92±8px, up-left/up-right 상단≤65px, down-left/down-right 상단≤13px. down의13px 기준은 topZ=-90의 실제 결과13/12에 맞춰 사용자가 명시 승인했다. 이전9/8px 관측은 topZ=-100일 때의 값이다.
- 피치6포즈의 방울 줄기 빈 틈≤15px. Fable과 같은 tie R>200/alpha>100 마스크의 x percentile20..80에서2px 간격으로 열을 택하고, 방울 직전 행부터 전경(crown|tails|face-skin)을 만날 때까지 빈 픽셀 수를 센다. 얼굴에 가려진 열을 빈 배경으로 세지 않는다. hairOnlyStemGap은 별도 보존했다. 중립 줄기값은 진단값이며 중립의 필수 계약은 픽셀0/정점해시 불변이다.
- 저자유도 계수/좌우 대칭/중력 방울/구면 이징/45px 상한 및 main89fc408 보호19파일 불변.

필수 지표 24개는 모두 통과했다. 보충0 대비 외곽 이동 최대 **18px ≤45px**다.

| 포즈 | 상단Y | 왼쪽 전경 줄기 틈 | 오른쪽 전경 줄기 틈 | 외곽 이동 최대 |
| :--- | ---: | ---: | ---: | ---: |
| up-left | 60 | 15 | 0 | 0 |
| up | 97 | 3 | 3 | 1 |
| up-right | 61 | 0 | 14 | 0 |
| left | 10 | 15 | 0 | 0 |
| front | 10 | 3 | 3 | 1 |
| right | 10 | 0 | 40 | 0 |
| down-left | 13 | 14 | 0 | 18 |
| down | 7 | 3 | 0 | 1 |
| down-right | 12 | 0 | 14 | 18 |

## 원화 대비 진단 잔차

엄격한 원화/SAM 일치는 별도 진단이다. 아래 20개 잔차를 rawResidualFailures에 유지했고 필수 게이트와 구분했다. 특히 up-right 상단의 참조48px 대비 차이와 down 정면의 기존 cropped≤4px 차이는 승인한 통과 범위와 서로 다른 지표다.

- up-left / hairline median: `12.5`
- up-left / forehead exposure: `0.10940868493994449`
- up-left / posterior exterior: `44`
- up / upper exterior: `86`
- up / hairline median: `-20.0`
- up / forehead exposure: `0.11308784093138868`
- up-right / crown top: `61`
- up-right / hairline median: `-39.0`
- up-right / posterior exterior: `72`
- down-left / crown top: `13`
- down-left / forehead exposure: `0.13484887625936448`
- down-left / posterior exterior: `70`
- down / crown top: `7`
- down / upper exterior: `43`
- down / forehead exposure: `-0.1463126843657817`
- down-right / crown top: `12`
- down-right / hairline median: `-13.5`
- down-right / forehead exposure: `-0.26193348268137173`
- down-right / posterior exterior: `129`
- down-right / near cheek background gap: `{"sampledRows": 61, "maximumGap": 1}`

## 설정·검증·캡처

설정 정본 source/viewer/hair-pitch.json과 생성 dome-fit.json/dome-data.mjs의 일치를 유지했다. topZ=-90/diagonalTopZ=-25, tipLag up=.35/down=.35/up-diag=.6/down-diag=.85, 양쪽 up rear0 및 양쪽 down rear 공통계수는 그대로다. tipLag를 y230에서 시작하는 이유는 앞머리 갈래 틈과 hairLift의 lowerMix가 y230부터 시작하기 때문이다.

- 제품 코드 Node 전체 **146 passed**, Python 전체 **37 passed**. 이후 승인된 측정 기준 정정은 영향 Python **14 passed**로 확인했다.
- 독립 증분 코드 감수 통과. 사용자에게 전달받은 Fable/세라피엘 재검수 결과를 유지하고 마지막 하향 대각선 회귀를 수정했다.
- 3시트를 직접 열어 위·아래·중립9포즈의 정수리와 방울 주변을 대조했다. up3/down정면/중립3의 PNG는 이전 실행과 바이트 동일하다.
- 요약 지표는 docs/hair-r15-metrics.json, 행별 원시는 git에서 제외되는 debug/hair_h0/current/hair-r15-metrics-raw.json에 저장한다. 원화/얼굴/렌더러/아틀라스 변경은 없다.
- captures/viewer_9dir_grid.png, captures/faceless_hair_9dir_comparison.png, captures/hair_only_9dir_overlay.png 및 아래 해시가 같은 단일 실행의 증거다. 머지는 담당 세션에 인계한다.

## SHA256

```text
ea46ddeb87634f7f2b63ee9fa3870677cbfc758519548d1b74a4c4fec6b67ab4  bundle/index.html
32a3cad1288a89dea6808c96b72a1c7c9ea4939865a74ab24835e28bc2e7963c  docs/captures/faceless_hair_9dir_comparison.png
4428ff633c9184b87e219f1dfaa24d2973b83d5e157b2d405e0c9cb40c56a61c  docs/captures/hair_only_9dir_overlay.png
d3f60b7e7c278d01fe34b83012002e1e32a9552edb3e29bce7e924e2a89ba1d5  docs/captures/viewer_9dir_grid.png
```
