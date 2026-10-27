# A1 후처리 재현

원래 A단계 생성 경로는 상위 README에 있습니다. 아래 코드는 승인 입력과 해를 보존한 후처리 경로이며 PR 검증에서 전체 솔브를 재실행하지 않았습니다.

```bash
python3 pipeline/face_outline_stage_a/refinement/refine_profile.py . .local/refined-keys .local/frozen-a
python3 pipeline/face_outline_stage_a/refinement/profile_b.py . .local/profile-b
python3 pipeline/face_outline_stage_a/refinement/profile_a.py . .local/profile-a
```

refine_profile은 먼저 B 경로를 복원하여 legacy 혼합이 앵커 복원을 가리지 않게 합니다. 실제 런타임 앵커를1e-12 아래로 닫고 누적보정1e-11 상한을 지킵니다. 9키 자유X투영은 앵커와샘플을 고정합니다. 원래 동결 manifest와 bundle은 history/f1a18d38에 보존합니다.

profile_b는 모든 교점 행을 새45 메시 외곽으로 통일하고 family45를 다시 풉니다. 67/90도 fixedSideNodes를 고정해 자유X만 폅니다. 전후du/이동목록은 accepted/family45-projection.json에 기록합니다. family45_expectations.mjs는 계약 기대값만 obs+저장이동으로 만들며 raw face_solve_inputs에는 되넣지 않습니다.

profile_a는6dd의 고정된9개 모듈과 절대 위치를45~67.5에서 혼합합니다. 강제 앵커는 legacy named anchor로 보내고 일반점은 일반경로를 유지합니다. 내부 pitch 폭에는 혼합을 중복 적용하지 않습니다. 마지막에 기존 X투영을 적용합니다. 투영 통계에는 새 문턱이 없습니다.

비용: 67.5°부터 6dd와 같아져 떠 있는 귀와 직선으로 잘린 볼이 돌아온다. 45~67.5°는 섞인다.

Node **207/207 pass**, fail0/cancelled0. 픽셀·투영 **16 passed**, 계약 **6/6**. 정면 전체 렌더 변경0픽셀.

상세 계약·SHA·시트는 accepted/final-validation.json에 있습니다. runtime stats는 원래 솔브 진단값이며 후처리 접힘 판정은 실제 런타임/Renderer 기하로 합니다. A2-1은 포함하지 않습니다.
