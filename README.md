# Seosoyoung Motion Portrait

서소영 모션 포트레이트의 3D 기준 형상과 2D 얼굴 변형을 연구하는 저장소입니다.

- [구면·타원체·원통의 실측 오차 비교](docs/surface-comparison.md)
- [Live2D 부위별 정점과 격자 측정](docs/measurements.md)
- [3D 회전 + 2D 잔차 보정 수식](docs/deformation-model.md)
- [의존성 없는 JS 모듈](src/head-turn.mjs)과 [서소영 좌표 연결 예제](examples/seosoyoung-head.mjs)

현재 산출물은 기하 프로토타입입니다. 외부 모델 파일과 서소영의 완성된 보정 키폼을 포함하지 않으며, 운영 뷰어에는 아직 연결하지 않았습니다.

## 실행

Node.js 22에서 확인했습니다. 모듈 자체에는 외부 의존성이 없습니다.

```sh
node --test --test-concurrency=1 --test-timeout=60000 tests/head-turn.test.mjs
node examples/seosoyoung-head.mjs
```

Python 분석 도구는 NumPy를, 도표 출력은 Matplotlib를 사용합니다. 분석 테스트는 다음과 같습니다.

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

## 측정 재현

인가받은 로컬 Core, MOC와 독립 추출 raw table을 입력으로 제공합니다. 샘플 파일은 이 저장소에서 배포하지 않습니다. 출력 정점과 fitted field는 공개 저장소 밖에 저장해야 합니다.

```sh
node tools/sample-core.cjs CORE.js MODEL.moc3 POSES.json PRIVATE_SAMPLES.json.gz
python3 tools/analyze_samples.py TABLES.json.gz PRIVATE_SAMPLES.json.gz REGIONS.json PRIVATE_ANALYSIS.json.gz
node tools/evaluate-residual.mjs PRIVATE_SAMPLES.json.gz REGIONS.json PRIVATE_ERRORS.json
python3 tools/fit_surfaces.py PRIVATE_SAMPLES.json.gz REGIONS.json PRIVATE_FIT.json
python3 tools/fit_surfaces.py PRIVATE_SAMPLES.json.gz REGIONS.json PRIVATE_FIXED_FIT.json --fixed-angles
```

`POSES.json`은 `{"xy_30_0":{"ParamAngleX":30,"ParamAngleY":0}}` 같은 포즈명→채널값 사전입니다. 분석/fit에는 두 축 `[-30,-15,0,15,30]`의 25개 `xy_X_Y` 포즈가 필요합니다. ordinary/roll 상세 분석에는 `standardX±30`, `standardY±30`, `standardZ±30`, `roll_±30`도 함께 생성합니다. sampler가 정면과 재평가 정면을 추가합니다. 직접 연결된 채널이 다른 모델은 실제 채널명을 사용해야 합니다.

`REGIONS.json`은 다음 형태입니다. `skin,nose,mouth,ears,eyes_left,eyes_right,eyeballs_left,eyeballs_right`는 표면 비교에 필요합니다. 머리카락과 목 비교에는 `hair_front,hair_side,hair_back,neck`을 추가합니다. 메쉬 ID와 포즈는 자기 모델에서 확인한 값을 씁니다.

```json
{
  "face": "skin_mesh_id",
  "regions": {"skin": ["skin_mesh_id"]},
  "channels": ["ParamAngleX", "ParamAngleY", "ParamAngleZ"],
  "extra_channels": []
}
```

raw table 입력은 `parameters,normal_parameter_bindings,normal_binding_bands,deformers,warps,warp_keyforms,position_floats,blend_keyform_bindings,blend_parameter_bindings,mesh_keyforms,rotation_keyforms,meshes` 구조를 가진 추출 결과입니다. 배열 인덱스와 좌표계를 검증한 추출기에서만 받습니다. 이 저장소는 MOC 파서나 Cubism 런타임을 재배포하지 않습니다.

## 공개 범위

일반 수학 코드, 직접 작성한 분석 도구, 집계 결과와 합성 테스트만 포함합니다. 외부 모델의 텍스처·바이너리·격자/메쉬 데이터·잔차 계수, 인증값과 내부 환경 경로는 포함하지 않습니다. 문서에 언급된 외부 모델의 권리는 각 제작자에게 있습니다.
