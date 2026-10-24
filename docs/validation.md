# Hair-only validation

현재 수식·계수·검증과 남은 잔차는 [최종 보고](hair-low-dof-report.md)와 [요약 지표](hair-r15-metrics.json)를 따른다. 이전 포즈별 타일 합성 증거는 제거했고, 마지막 코드에서 실행한 단일 H0 캡처로 갱신했다.

- 설정 정본: [hair-pitch.json](../source/viewer/hair-pitch.json)
- 전체 Node144 / Python34, 최종 영향 Node16 / 측정기11 passed
- 보호 기준: main89fc408, [19파일 SHA256](hair-hspec-protected-sha256.txt)
- 최종 HTML/3시트 해시: [SHA256](hair-low-dof-sha256.txt)
- [전체9방향](captures/viewer_9dir_grid.png), [Hair-Only 대조](captures/faceless_hair_9dir_comparison.png), [오버레이](captures/hair_only_9dir_overlay.png)

얼굴·피처·렌더러·원화·아틀라스는 보호 대상으로 유지했다. 코드 감수와 실제 캡처 판정, 전문 UI/UX 검수 및 머지를 구분한다. 원화/SAM 정합의 엄격한 게이트와 상한 준수에 따른 bounded pass는 최종 보고에 각각 기록했다.
