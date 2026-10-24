# Hair-only validation

[최종 보고](hair-low-dof-report.md)와 [요약 지표](hair-r15-metrics.json)는 단일 H0 캡처를 사용자 승인 기준으로 재집계한 bounded pass를 기록한다. 추가 캡처는 없으며 필수 상단·전경 줄기·중립 불변 게이트와 원화 대비 진단 잔차를 구분한다.

- 전체 Node146 / Python37, 이후 측정기 영향 Python14 passed
- 설정 정본: [hair-pitch.json](../source/viewer/hair-pitch.json)
- 보호19파일: main89fc408 및 [SHA256](hair-hspec-protected-sha256.txt)
- HTML/3시트 해시: [SHA256](hair-low-dof-sha256.txt)
- [전체9방향](captures/viewer_9dir_grid.png), [Hair-Only 대조](captures/faceless_hair_9dir_comparison.png), [오버레이](captures/hair_only_9dir_overlay.png)

얼굴·피처·렌더러·원화·아틀라스는 보호 대상으로 유지했다. 원시 행별 데이터는 debug/에만 저장하고 이전 포즈의 캡처 합성은 사용하지 않았다.
