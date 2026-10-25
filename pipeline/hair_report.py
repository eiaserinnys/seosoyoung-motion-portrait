"""Comparison sheets and compact release evidence from raw hair measurements."""
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def sheets(rows, out):
    comparison = Image.new('RGB', (1080, len(rows) * 360), '#182330')
    overlay_sheet = Image.new('RGB', (1080, 1080), '#182330')
    portrait_sheet = Image.new('RGB', (1440, 1440), '#24303c')
    for i, row in enumerate(rows):
        ref, render, mask = row['ref'], row['render'], row['mask']
        portrait = Image.new('RGBA', (1024, 740), '#24303c')
        portrait.alpha_composite(Image.fromarray(render))
        tile = portrait.crop((152, 20, 872, 740)).convert('RGB').resize((480, 480))
        ImageDraw.Draw(tile).text((8, 8), row['name'], fill='white')
        portrait_sheet.paste(tile, ((i % 3) * 480, (i // 3) * 480))
        reference_hair = ref['rgb'].copy()
        reference_hair[~ref['mask']] = [24, 35, 48]
        actual = render[:, :, :3].copy()
        actual[~mask] = [24, 35, 48]
        over = np.zeros_like(actual) + [24, 35, 48]
        over[ref['mask']] = [40, 170, 220]
        over[mask] = [240, 110, 80]
        over[mask & ref['mask']] = [210, 210, 210]
        for j, (image, title) in enumerate([(reference_hair, 'reference'), (actual, 'render'), (over, 'overlay')]):
            tile = Image.fromarray(image.astype(np.uint8)).crop((152, 0, 872, 720)).resize((360, 360))
            ImageDraw.Draw(tile).text((6, 6), row['name'] + ' ' + title, fill='white')
            comparison.paste(tile, (j * 360, i * 360))
            if j == 2:
                overlay_sheet.paste(tile, ((i % 3) * 360, (i // 3) * 360))
    comparison.save(out / 'faceless_hair_9dir_comparison.png')
    overlay_sheet.save(out / 'hair_only_9dir_overlay.png')
    portrait_sheet.save(out / 'viewer_9dir_grid.png')


def summarize_metrics(data):
    """Keep release evidence small; full per-row samples belong in debug/."""
    result = json.loads(json.dumps(data))
    for p in result['poses']:
        exterior = p['exterior']
        p['exterior'] = {'sampledRows': len(exterior), 'maxAbsoluteError': max(
            (abs(v) for r in exterior if r['delta'] for v in r['delta']), default=None)}
        shifts = p['posteriorProjectionShift']
        worst = max((r for r in shifts if r['shift']), key=lambda r: max(abs(v) for v in r['shift']), default=None)
        p['posteriorProjectionShift'] = {'sampledRows': len(shifts), 'worstRow': worst,
                                       'maximum': max(abs(v) for v in worst['shift']) if worst else None}
        cover = p['nearCheekCover']
        p['nearCheekCover'] = {'sampledRows': len(cover), 'maximumGap': max((r['gap'] for r in cover), default=None)} if cover is not None else None
        p['hairline'] = {'sampleCount': len(p['hairline']), 'medianError': p['hairlineMedianError']}
    for c in result['checks'] + result.get('rawResidualFailures', []):
        if c['gate'] == 'near cheek background gap':
            rows = c['value'] or []
            c['value'] = {'sampledRows': len(rows), 'maximumGap': max((r['gap'] for r in rows), default=None)}
    return result


def write_release_report(root, output, data):
    out = root / output
    perf = json.loads((root / 'docs/hair-performance.json').read_text())
    p = perf['finalReducedCacheComparison']
    lines = ['# 상향 대각 앞머리·후두부 볼륨 검증', '',
             f"실측 판정: **{data['acceptance']}**. 기준은 `{data['baseline']}`이며, 모든 비교 시트는 같은 최종 9포즈 캡처에서 생성했다.", '',
             '## 이전 프레임 비용 검증과 보호 범위', '',
             f"정점 21,855 → {p['vertices']:,}, 삼각형 35,976 → {perf['finalTriangles']:,}. 고밀도 P1/P2는 4,102.99 → 2,039.96ms, 감축 메시의 동일 수식 비교는 {p['beforeMs']:.2f} → {p['afterMs']:.2f}ms다. 9포즈 최대 정점 변위는 모두 0.000px다. 이는 Node 변형 시간이며 브라우저 FPS 측정은 아니다.", '',
             f"보호 파일 {data['protectedFileCount']}개: {data['protectedFilesPass']}. 별도 계약 테스트는 모든 비머리카락 메시와 rig 메타데이터, 전체 parts 및 texture_00의 기준 커밋 불변을 검사한다.", '',
             'P1/P2는 행별 재계산·메시 체인 보간·마스크 중복 변형을 제거했다. T1/T2는 꺼져 있던 45° 참조 RGB 혼합을 복구했다. V1/V2는 승인된 대각 상단 깊이와 후두부 계수 및 상단 감쇠다. P3는 canopy step12, cranium16×48, 후면/꼬리 simplify3.0·area800·seam6칸 간격을 쓴다.', '',
             '## 방향별 실측', '',
             '| 포즈 | 상단Y | 림라이트 감소 | 하단 후두부 평균 오차 전 → 후 | P3 최대 p95 | 볼 틈 |',
             '| :--- | ---: | ---: | ---: | ---: | ---: |']
    for pose in data['poses']:
        rim, volume = pose['templeRimlight'], pose['posteriorVolume']
        reduction = f"{rim['reduction']*100:.2f}%" if rim and rim['reduction'] is not None else '해당 없음'
        posterior = f"{volume['baselineMeanAbsoluteDelta']:.2f} → {volume['meanAbsoluteDelta']:.2f}" if volume else '해당 없음'
        p95 = max(s['p95'] for s in pose['retriangulationSilhouette'].values())
        cover = pose['nearCheekCover']
        gap = cover['maximumGap'] if cover else '해당 없음'
        lines.append(f"| {pose['name']} | {pose['topY']} | {reduction} | {posterior} | {p95:.2f} | {gap} |")
    failed = [c for c in data['invariants'] if not c['pass']]
    lines += ['', '## 필수 게이트', '', json.dumps(failed, ensure_ascii=False) if failed else '모든 필수 게이트 통과.', '',
              'B1+B2+B3 승인 기준: 좌상·우상 이마 노출 면적비 오차15% 이내(그 외10%), 후두부 활성 행340..440의 평균 절대 오차는 좌상16px·우상20px 이내다. 기존 평균 오차 개선10px 이상과 상단120..200의 양의 돌출 증가13px 이내도 유지한다. rearPitch와 rearLower를 함께 끈 stage a 대비 래스터 외곽 이동은60px 이내다. B2로 정상 앞머리 하이라이트가 측두부 창에 들어오는 효과를 반영해 림라이트 감소50% 이상을 승인했다. 하향 볼 배경 틈2px 이내, 방울 줄기17px 이내 기준을 유지한다.', '',
              'B1은 좌상·우상 앞머리 끝 피치 추종률을0.85로 높인다. B2는 상향 대각선의 시선 방향 앞머리를 바깥으로 쓸고, B3는 하단 후두부를 방사 확장하며 상향 회전의 X를 보존한다. 설정 정본은 source/viewer/hair-pitch.json이다.', '',
              f"측정 모드: {data['measurementMode']}. 제품 파일 해시와 측정기 해시는 hair-r15-metrics.json에 기록한다. 원화 대비 엄격한 잔차는 진단값으로 보존한다.", '',
              'P3는 같은 새 볼륨 수식의 고밀도 토폴로지와 비교하고, 중립 3포즈는 실제 이전 렌더와도 비교한다. 원화 대비 엄격한 잔차는 hair-r15-metrics.json의 rawResidualFailures에 보존한다. UI 마크업·컨트롤·텍스트·레이아웃 변경은 없다. 전문 UI/UX 검수와 머지는 담당 세션에 인계한다.', '']
    (out / 'hair-low-dof-report.md').write_text('\n'.join(lines))
    paths = [root / 'bundle/index.html', *[out / 'captures' / name for name in ['faceless_hair_9dir_comparison.png', 'hair_only_9dir_overlay.png', 'viewer_9dir_grid.png']]]
    (out / 'hair-low-dof-sha256.txt').write_text(''.join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n" for path in paths))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    write_release_report(root, 'docs', json.loads((root / 'docs/hair-r15-metrics.json').read_text()))
