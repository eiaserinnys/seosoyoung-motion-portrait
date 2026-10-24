# Hair depth and attachments on the r15 baseline

PR #8 is based on `d6a58ba5d0b409389ac1baf4ffdae6a0695c5364`. The face, facial features, renderer, side strands and rig retain that baseline byte for byte. The 19-file manifest is [hair-r15-protected-sha256.txt](hair-r15-protected-sha256.txt). The separate `earWorld` implementation is also unchanged.

## Hair geometry

The existing 45-degree crown, strand and bead fits constrain the absolute depth. A cached Depth Anything V2 Small inference contributes internal crown relief; ordinary fitting uses the checked-in cache without model dependencies. Its model revision, source image digest and calibration are recorded in [hair-depth-prior.json](../source/viewer/hair-depth-prior.json) and [dome-fit.json](../source/viewer/dome-fit.json).

The new rotation and six-reference residual act inside the crown. Their envelope is zero on the approved perimeter and below source Y310. Rear cranium and backing use the r15 reference surface throughout. Removing the extra downward horizon turn, rear seam transfer and positive crop-width expansion eliminates the inflated down-diagonal helmet, brown patch bands and up-diagonal step notches. No texture, alpha, atlas, face or side-strand changes are included.

## Far upward tie attachment

In r15 the upward far tie and tail remain below the lifted skull. Only positive pitch and the far side receive the shared lower-rim Y bound, smoothly weighted by yaw. The original bead rotation remains intact for neutral, downward, frontal and near-side poses. Both tail layers inherit the corrected root. The frame cache stores an unrolled scalar displacement; roll rotates that displacement after lookup. Ear geometry is unchanged.

Measured blue tie coordinates remain observations. Their biased facial-registration offsets are not applied to cheek locks or tail roots. The C correction leaves the approved A/B crown and rear geometry unchanged.

## Verification

| Gate | Result |
|:--|:--|
| Node tests | 118 passed, 0 failed, including all 12 face-registration tests |
| Python tests | 16 passed |
| Outline raster | 22 poses passed; every pitched cheek/ear gap count is 0 |
| Geometry preservation | Nine-direction vertex hashes preserve the r15 faces, rear surfaces, lower side locks and attachments, except the two authorized far/up tail corrections |
| Far/up attachment | Left/right tie overlap is 33.8094/32.0265 px; required minimum is 25 px |
| Full viewer | PASS; canopy 972/error 0, console/requests/WebGL errors 0, clipping/controls/auto/lightbox/narrow layout passed |
| Pixel preservation | Pitch-zero middle row equals r15 exactly; down row equals the approved A/B capture exactly |
| Protection | 19 baseline files and the original `earWorld` implementation are byte identical |

Full browser metrics and the refreshed nine-direction capture are linked below. Browser verification covers the actual WebGL entry point, canopy continuity, attachment geometry, feature clipping and visibility, controls, auto motion, lightbox and narrow layout. Projected back faces remain permitted and culled; 3D collapse checks remain active.

Evidence: [nine directions](captures/viewer_9dir_grid.png), [hair metrics](hair-r15-metrics.json), [viewer metrics](viewer_metrics.json), [outline metrics](outline_metrics.json), [pitched gaps](pitched_outline_metrics.json), [bundle SHA256](bundle-sha256.txt). Independent Sol 6.1 review passed on the source and final nine-direction capture with no P0/P1 findings. Direct comparison with r15 and the original up/down diagonal references shows a smooth rear silhouette without the inflated helmet, brown patch band or step notch; both upward far ties now join the lower hair. The face and near side-hair curves remain the r15 curves. Specialist UI/UX review, merge and publication remain with the coordinator.


## H0/H1/H2/H4/H5/H6 저자유도 구면 보정 (261025)

origin/main 89fc408을 현재 브랜치에 머지했다(47d441d). 이전 미커밋 피팅 작업은 stash 537f47b73897725ed1125495c59b24bbb8df7080에 보존했고, fit_hair_pitch.py/fit_hair_raster.py/fit_hair_tip.py 및 그 자유계수는 최종 작업 트리에서 제거했다. 아래 결과가 이전 Hspec 실패 및 r15 내부 다변수 잔차 설명을 대체한다.

H1은 topZ=-100 한 개의 원호와 승인된 극점 수렴 w=cos(lambda)+(1-cos(lambda))*sphereEase(s_y)를 사용한다. fringeSurface의 X/Z는 항상 hairLift의 artWorld 값과 동일하다. H2는 15도 collar와 실제 피치 회전을 사용하며, 대각선마다 rear rx/rz/y 세 계수만 s_rear*cos(phi)에 곱한다. H4는 방울을 yaw 중력 프레임에 두고 옆머리 y405..544를 연결한다. H5는 추가 승인에 따라 y230..420에서 포즈당 kappa 한 개를 적용한다. 선택값은 up-left .6, up .35, up-right .35, down-left .85, down .35, down-right .6이다. H6은 pitchEase를 smooth+(sphere-smooth)*sphereEase(abs(pitch)/30)로 연결한다.

독립 코드 감수에서 발견한 중립 후두부 seam 점프를 수정했다. 중립 legacy-vs-raw 차이를 (1-sphereEase(abs(pitch)/30))로 소거하여 중립 XYZ를 보존하고 30도에서 실제 3D를 그대로 사용한다. 변경 전 회귀 1 fail, 변경 후 최종 영향 테스트 43 passed. 독립 재검수의 가시 near-side 재현점 점프는 약3.1e-8px이며 추가 반려 없음이다.

검증 결과:

- Node 전체 137 passed / 0 failed. 이후 연속성 변경은 영향 범위 43 passed / 0 failed로 최종 확인했다. 전체 스위트 2회 상한을 지켰다.
- Python 전체 32 passed. 이후 측정기의 누락행 JSON 직렬화와 그 회귀 데이터는 최종 영향 범위 9 passed로 확인했다. 전체 스위트 2회 상한을 지켰다.
- origin/main 대비 보호 파일19개와 혼합 테스트의 얼굴/피처 본문13개 byte-identical. 중립 left/front/right는 픽셀0 및 전체 정점/craniumPosition/craniumFacing/domeSection 해시 일치.
- H1 X/Z 최대 오차0px. 방울 pitch 불변 오차0px. 피치6포즈 양쪽 줄기 틈0px.
- 후두부 추가 변위 최대 표본41.148692px, 같은 최종 상태에서 보충0으로 렌더한 stage(a) 대비 모든 공통행80..520 투영 외곽 이동 최대44px. 상한45px 이내다.
- 최종 H0는 사용자 승인 정책에 따른 bounded pass, exit0이다. 엄격한 원화 정합은19조건 미달이며 원시값을 hair-r15-metrics.json에 보존했다. 누락된 외곽 행은 JSON null 및 사유로 명시한다.

실화면은 요청한 전체 초상9포즈, 머리카락 단독 참조 대조9포즈, 오버레이9포즈 3시트를 모두 직접 열었다. 이전 음수 깊이의 X/Z 혼입으로 발생한 약280px 수평 전단과 뾰족한 상단 찢어짐, 후두부 풍선 팽창은 사라졌다. up 정수리의 평탄한 윤곽과 일부 대각선 결 늘어짐/실루엣 차이는 남아 있다. UI 마크업/글자/간격/컴포넌트는 변경하지 않았다.

남는 수치: top up102px/down2px/down-left9px/down-right8px. 헤어라인 median up-20px/up-right-33px/down+10px/down-right-20px. 후두부 원화 외곽 잔차 최대 up-right54px. down-right의 가까운 하관에는 실제 래스터 배경1px 틈이 y514..516/518에 남는다(원화 대비 오차와 구분). 이 원시 게이트를 strict pass로 표기하지 않는다. 별도 전문 UI/UX 검수와 PR 머지/발행은 담당 세션에 인계한다.

범위 열거: 공통 앞머리 표면을 쓰는 hair-top/bangs-left/bangs-right/hairpin/side-hair-left/side-hair-right, backing3개, cranium2개, tail hair2개와 tie2개를 측정했다. 양쪽 귀 함수 본문과 얼굴/피처/렌더러/아틀라스/원화는 보호 대상이다. 런타임 변경은 네 머리카락 모듈과 그 DOME 데이터에 한정했다. 요청 밖의 렌더러/UV/face 보정과 추가 자유도는 구현하지 않았다.

증거는 hair-r15-metrics.json, hair-stage-a-metrics.json, hair-low-dof-calibration.json, hair-hspec-protected-sha256.txt, hair-low-dof-sha256.txt 및 captures의 3시트다. metrics의 head는 캡처 당시 기반 HEAD이고, productSha256이 최종 제품 코드와 캡처를 연결한다. 코드/부착/중립 테스트와 실화면 수용을 구분한다.


## 상향 정수리 볼록성과 후두부 Y 플랩 교정 (261025)

사용자의 캡처 지적과 확정 수식으로 c8b358f의 상단 기하를 교정했다. 상수 z310과 경도 감쇠가 옆쪽 깊이를 앞쪽으로 뒤집던 원인은 각 점의 correctedLift 깊이에서 topZ로 보간하여 제거했다. y310의 열별 C1 접속과 y8의 공통 극점 수렴을 함께 유지한다.

후두부 Y는 fringeSurface(edge)와 canopyWorld(edge)의 crownBlend 높이를 공유하고 posterior 보충 차분만 더한다. 깊은 두개골의 raw pitch Y와 15도 Y collar는 제거했다. X/Z의 3D 좌표, collar 접합 및 중립 C1 소거는 유지했다. craniumPosition의 includePosterior=false 경로로 같은 표면의 보충0 값을 명시적으로 구한다.

회귀 2개 RED→GREEN. Node 전체140 passed, Python 전체32 passed. 독립 코드 감수 통과이며 y310 접속 오차는 약1.7e-8px다. 얼굴/렌더러/아틀라스/원화19파일과 얼굴피처 단언13본문은 main89fc408과 byte-identical, 중립3포즈pixel0 및 정점/후두부/단면 해시 일치다.

후두부 4대각선의 세 계수를 보충0 캡처에서 다시 산출했다. 마지막 전체 H0는 exit1/46px였으며, 사용자 승인에 따라 up-right 보충0 실측과 다른8포즈 실측을 재사용한 최종 판정이 bounded pass/40px다. 추가 H0 실행은 없었다. 앞머리 지체 계수와 중력 프레임은 유지했다. 원시 참조/마스크 게이트 미달 22개는 hair-r15-metrics.json에 그대로 기록했다.

최종3시트를 직접 열어 up의 넓은 평평한 지붕/좌우 뿔과 up-left/up-right 귀 뒤의 긴 갈색 삼각 플랩이 해소됐음을 확인했다. 새 공식은 down-right의 raw posterior pitch Y 과솟음도 제거한다. 머리카락 단독 참조 및 오버레이로 상단과 하단을 함께 비교했다. 이전 문단의 up 평탄화 미해결 판정은 이 수정으로 대체한다. 수치 정본은 최종metrics/새 stage-a/갱신calibration, 캡처와 HTML 해시는 hair-low-dof-sha256.txt다.


최종 설정과 증거 출처는 `hair-low-dof-report.md`에 정리했다. 수동 설정 정본은 `config/rig-config.json`, 생성 데이터는 `source/viewer/dome-fit.json` 및 `bundle/dome-data.mjs`이며 up-right는 rx/rz/y 모두0이다. 최종 관련 Node14개가 통과했고 독립 증분 검수도 통과했다.
