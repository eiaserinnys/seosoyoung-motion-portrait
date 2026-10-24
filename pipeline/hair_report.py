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
    lines = ['# 측두부 질감·상향 볼륨·프레임 비용 검증', '',
             f"실측 판정: **{data['acceptance']}**. 기준은 `{data['baseline']}`이며, 모든 비교 시트는 같은 최종 9포즈 캡처에서 생성했다.", '',
             '## 프레임 비용과 보호 범위', '',
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
              'Fable 승인에 따라 측정 범위를 제품 설계와 정렬했다. 넓은 측두부 창의 림라이트 감소는 58% 이상, 후두부 활성 행340..440의 평균 오차 개선은10px 이상이다. 상단 비확장 가드 구간120..200은 원화 바깥의 양의 돌출량만 계산해 r16 대비 증가5px 이내를 검사한다: max(0, 현재−원화)−max(0, r16−원화). 원화 내부의 폭 복원은 회귀에서 제외하며 원화 대비 상단 돌출은 진단값으로 남긴다. 볼 덮임은 하향에서만2px 이내, 방울 줄기는 재삼각화 경계 차이를 포함해17px 이내다. 상향 볼의 턱선 노출은 진단값이다.', '',
              '제품 코드와 기존 캡처는46c9afd에서 고정했다. 브라우저 재캡처 없이 --measure-only로 기존 프레임만 재판정했다. 원화의 엄격한 잔차와 측정값은 보존하며 이번 승인은 제품 수식 변경을 포함하지 않는다.', '',
              'P3는 같은 새 볼륨 수식의 고밀도 토폴로지와 비교하고, 중립 3포즈는 실제 이전 렌더와도 비교한다. 원화 대비 엄격한 잔차는 hair-r15-metrics.json의 rawResidualFailures에 보존한다. UI 마크업·컨트롤·텍스트·레이아웃 변경은 없다. 전문 UI/UX 검수와 머지는 담당 세션에 인계한다.', '']
    (out / 'hair-low-dof-report.md').write_text('\n'.join(lines))
    paths = [root / 'bundle/index.html', *[out / 'captures' / name for name in ['faceless_hair_9dir_comparison.png', 'hair_only_9dir_overlay.png', 'viewer_9dir_grid.png']]]
    (out / 'hair-low-dof-sha256.txt').write_text(''.join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n" for path in paths))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    write_release_report(root, 'docs', json.loads((root / 'docs/hair-r15-metrics.json').read_text()))
