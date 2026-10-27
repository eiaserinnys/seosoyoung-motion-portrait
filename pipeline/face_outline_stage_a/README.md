# 얼굴 외곽선과 귀 — 승인 A단계

출발 해는7.11-filled + 귀 이동 범위80이다. accepted/manifest.json은 원래 동결 해의 출처를 보존한다. 최종 제품 SHA는 accepted/final-validation.json에 있으며, 앵커 닫기·9키 자유X투영·프로필 후처리를 반영했다. O2 옆머리 실험과 눈PR24는 포함하지 않는다.

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

## 최종 검증

Node **207/207 pass**, fail0/cancelled0. 픽셀·투영 **16 passed**, 계약 **6/6**. 정면 전체 렌더 변경0픽셀.

- 9키와 키사이8장의 피부 음수 삼각형0, ±45 연속.01 유지.
- 혼합 전 B의 관측·행 합성은 승인된 오프라인 펴기 기록을 포함해1e-6으로 검사한다. 최종 A는67.5이상에서6dd와1e-12, 혼합구간에서Y/Z혼합과X혼합+동일프레임투영을1e-6으로 검사한다.
- up턱 기대값은 레퍼런스28px에서 측정 F2오차7.847817px를 뺀 값이며, 기존1e-12 앵커 기준은 유지했다.
- 상세 수치·SHA·시트 해시는 accepted/final-validation.json, 후처리 재현은 refinement/README.md.

## 알려진 비용과 범위

67.5°부터 6dd와 같아져 떠 있는 귀와 직선으로 잘린 볼이 돌아온다. 45~67.5°는 섞인다.

up-left 귀가 약간 높은 점은 남아 있다. 옆머리 줄무늬와 먼 가닥「<」는 O2 후속이다. A2-1과 눈PR24는 포함하지 않았다. 원래 동결 해는 그대로 보존한다.

기존 pipeline/solve_to_bundle.py는A단계 이전 source JSON을 내보내므로 이 승인 해를 보존할 때 덮어쓰지 않는다. 이 PR에서는 원래 source JSON을 변경하지 않았다.
