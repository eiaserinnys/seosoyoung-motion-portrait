# 얼굴 외곽선과 귀 — 승인 A단계

승인 해는 7.11-filled + 귀 이동 범위 80이다. `accepted/manifest.json`의 세 파일 SHA가 제품 `bundle/face-solve-data.mjs`, `stereo-attachments.mjs`, `ear-transform-data.mjs`의 기준이다. O2 옆머리 실험과 눈 정합은 포함하지 않는다.

## 입력과 생성 경로

`tps-study/stage-a/inputs/<pose>/`에 등록 프레임, 승인된 피부·귀 마스크, 머리카락 띠와 manifest가 있다. `tps/`는 최종 TPS다. 이 PR에서는 솔브·귀 적합·TPS를 다시 실행하지 않았다.

| 단계 | 보존한 생성 코드·데이터 |
|:--|:--|
| 등록·초기 입력 | `stage-a/input_masks.py`의 `prepare_pose`; 기존 `pipeline/ponytail_mesh.py` 등록 행렬 |
| SAM 3 | `sam3-c1/sam_requests.py`, `point-c1/requests_points.py`; 원본 출력 `*/raw/`. 최종 얼굴은 점 프롬프트, 귀 보임은 담당이 확인한 표다. 기존 fal/Vault 로더를 사용한다. |
| 승인 마스크 조합 | `point-c1/build_inputs.py` → `correct-c1/masks.py` → `hull-c1/hull_masks.py`. left는 `tps-study/v2/`의 검증된 입력을 재사용한다. |
| 내부 구멍·TPS | `v711-filled/fill_masks.py`; `fit_inputs.py`, `tps_distance_inputs.py`, `tps_distance.py`. 외부 윤곽은 유지하고 눈·입 내부 구멍을 채운다. |
| 관측 pin | `v711-filled/build_inputs.py`, `policy.py`, `hair_mass.py`. 경계 거리·실제 머리카락 덩어리·귀 상자로 관측을 구분한다. |
| A 솔브 | `v711-filled/stage_a.py`, `solve_observed.py`; 저장된 `runtime-projection.json`, `runtime-current.json`, `base-inputs.json`을 사용한다. |
| 귀·활성 집합 | `v711-filled/fit_ears.py`, `active.py`, `final_c2.py` → `v711-ear80/refit/fit_ears.py`, `merge_fit.py`, `update_active.py` |

새 마스크를 얻는 SAM 호출은 별도 외부 작업이며 자동으로 승인된 입력을 대체하지 않는다. 저장된 승인 입력으로 계산을 재현할 때는 아래 순서만 사용한다. 출력은 기존 하니스와 같이 해당 단계 폴더와 제품 `bundle/`다. `before-inputs`, `hair-mass`, `ears` 같은 실행 출력은 새 실행 폴더에서 생성한다. 역사적 runner 횟수·C1 승인 대기를 담은 batch 코드는 옮기지 않았다.

## 계산 순서

리포 루트를 기준으로 실행한다. 무거운 명령은 운영 환경의 heavy runner 안에서 한 번에 하나씩 실행한다. 이 목록은 실행 순서를 보존한 것이며 PR 검증에서 재실행하지 않았다.

```bash
python3 pipeline/face_outline_stage_a/tps-study/stage-a/v711-filled/fill_masks.py
python3 pipeline/face_outline_stage_a/tps-study/stage-a/v711-filled/build_inputs.py
python3 pipeline/face_outline_stage_a/tps-study/stage-a/v711-filled/solve_observed.py
node pipeline/face_outline_stage_a/tps-study/stage-a/v711-filled/export_bases.mjs pipeline/face_outline_stage_a/tps-study/stage-a/v711-filled .
python3 pipeline/face_outline_stage_a/tps-study/stage-a/v711-filled/fit_ears.py
python3 pipeline/face_outline_stage_a/tps-study/stage-a/v711-filled/final_c2.py
node pipeline/face_outline_stage_a/tps-study/stage-a/v711-ear80/refit/export_bases.mjs pipeline/face_outline_stage_a/tps-study/stage-a/v711-ear80/refit .
cp pipeline/face_outline_stage_a/tps-study/stage-a/v711-ear80/refit/ear-bases.json pipeline/face_outline_stage_a/tps-study/stage-a/v711-ear80/ear-bases.json
python3 pipeline/face_outline_stage_a/tps-study/stage-a/v711-ear80/refit/fit_ears.py
python3 pipeline/face_outline_stage_a/tps-study/stage-a/v711-ear80/merge_fit.py
python3 pipeline/face_outline_stage_a/tps-study/stage-a/v711-ear80/update_active.py
```

`export_bases.mjs`는 `earBase + t_face`와 실제 런타임의 합성을 검산한다. 80px 단계는 `selection.json`의 여섯 귀만 다시 맞추고, 바뀐 포즈의 du만 원래 관측에서 복원한다. Z와 얼굴 t는 고정한다. 가중치·λ·귀 크기·회전 범위는 바꾸지 않았다. `provenance.json`에 옮기기 전후 코드 SHA와 경로를 보존했다.

## 검증과 증거

- `pipeline/test_face_outline_stage_a.mjs`: 승인 당시 실제 렌더가 기록한 9키 + 키 사이 8장의 피부·귀 좌표와 반전 집합. 알려진 키 반전 11개는 같은 ID만 허용하며 중간 8장은 음수 0이다.
- `pipeline/test_face_outline_stage_a.py`: 같은 현재 뷰어·아틀라스에서 `seed-bundle/`의 변경 전 얼굴·귀 모듈과 현재 모듈을 불러 정면 전체 렌더를 대조한다. 0픽셀이 기준이다.
- 기존 r11/r12 위치·귀 상자·눈 여백은 승인 A 데이터로 기준을 갱신한다. 45도 밖 프로필 실패는 기준을 바꾸지 않고 진단 중이다.
- 시트: `docs/captures/face-outline-stage-a/`의 9키 머리카락 없음·켬, 키 사이 두 장, F7 세 자리.

남은 것: up-left 오른쪽 귀가 약간 높음(IoU .472, 이동 80 경계), 옆머리 줄무늬와 먼 「<」는 O2 재설계 대상이다. 행 F3 전행 정의는 v7 자기 검산에 실패해 판정에 쓰지 않았다. 향후 화면 `y ≥ y_c`로 제한해야 한다는 결정을 기록하며, 승인된 키 사이 틈 판정은 담당의 시각 확인이다.

기존 `pipeline/solve_to_bundle.py`는 A단계 이전 `source/viewer/face-solve.json`을 내보내는 별도 생성 경로다. 승인 A 데이터를 보존할 때는 이 옛 경로로 덮지 않고 위 A단계 경로를 사용한다. 이 이관에서는 기존 source JSON과 생성기를 변경하지 않았다.

## A1 제품 통합 현재 상태

정면 픽셀·머리카락 픽셀 14개와 승인 17포즈 기하 계약은 통과했다. 전체 Node 검사는 177 통과, 6 실패, 3 취소다. 45도 밖 프로필의 경계 점프·피부 반전·이마 간격 퇴보는 기존 게이트 그대로 보존해 원인 설계에 넘긴다. up 턱 곡률 옛 하한과 특징점 1e-12 비교도 미통과이며, 성능·포니테일·side90 파일은 60초 제한으로 취소됐다. 프로필 해결 전에는 A1을 머지하지 않는다.
