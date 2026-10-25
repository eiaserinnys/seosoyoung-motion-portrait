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


def before_after_sheet(root,out,poses):
    sheet=Image.new('RGB',(1920,960),'#24303c')
    for i,p in enumerate(poses):
        for col,(folder,label) in enumerate([('debug/hair_solve_prior','prior'),('debug/hair_h0/current','solve')]):
            rgba=Image.open(root/folder/(p['name']+'-full.png')).convert('RGBA')
            tile=Image.new('RGBA',rgba.size,'#24303c');tile.alpha_composite(rgba)
            tile=tile.convert('RGB').crop((152,20,872,740)).resize((320,320))
            ImageDraw.Draw(tile).text((8,8),p['name']+' | '+label,fill='white')
            sheet.paste(tile,((i%3)*640+col*320,(i//3)*320))
    sheet.save(out/'r15_before_after_9dir.png')


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
    for c in result['checks'] + result.get('rawResidualFailures', []) + result.get('legacyChecks', []):
        if c['gate'] == 'near cheek background gap':
            rows = c['value'] or []
            c['value'] = {'sampledRows': len(rows), 'maximumGap': max((r['gap'] for r in rows), default=None)}
    return result


def write_release_report(root, output, data):
    out = root / output
    solve=json.loads((root/'source/viewer/hair-solve.json').read_text())
    fits={p['name']:p for p in solve['poses']}
    charts={p['name']:p for p in data['hairSolveChart']['poses']}
    lines=['# 우하 얼굴 정본화와 머리카락 합동 피팅', '',
           f"비악화 게이트 판정: **{data['acceptance']}**. 3D 사전은 `{data['baseline']}`와 바이트가 같다. 절대 원화 목표 전부 통과: {data['strictRasterPass']}.", '',
           f"2D 차트: {data['hairSolveChart']['pass']}; 잔차 예산: {data['hairSolveBudgetPass']}; 중립 머리카락 정점 불변: {data['neutralGeometryPass']}.", '',
           '## 방향별 실측', '',
           '| 포즈 | 상단Y | 앞머리 오차 | 이마 면적비 오차 | 볼 틈 | 잔차 RMS / 최대 | 차트 minDet |',
           '| :--- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for pose in data['poses']:
        name=pose['name'];fit=fits.get(name);chart=charts.get(name)
        cover=pose['nearCheekCover'];gap=cover['maximumGap'] if cover else '해당 없음'
        residual=f"{fit['stats']['duRms']:.2f} / {fit['stats']['duMax']:.2f}" if fit else '0 / 0'
        determinant=f"{chart['minDet']:.4f}" if chart else '1'
        ratio=pose['foreheadExposure']['ratioError'] if pose['foreheadExposure'] else None
        lines.append(f"| {name} | {pose['topY']} | {pose['hairlineMedianError']} | {ratio} | {gap} | {residual} | {determinant} |")
    lines+=['','## 동일 사전 대비 실제 렌더 오차','',data['priorComparison']['definition'],'',
            '| 포즈 | 사전 RMS | 보정 RMS | 판정 |','| :--- | ---: | ---: | :--- |']
    for p in data['priorComparison']['poses']:
        lines.append(f"| {p['name']} | {p['priorRms']:.3f} | {p['correctedRms']:.3f} | {'통과' if p['pass'] else '실패'} |")
    failed=[dict(pose=c.get('pose'),gate=c['gate'],value=c.get('value')) for c in data['checks'] if not c['pass']]
    lines+=['','## 남은 원화 목표 오차','',json.dumps(failed,ensure_ascii=False,indent=2) if failed else '목표 게이트 통과.','',
            '## 재생성','',
            '얼굴: fit_features.py → fit_outline.py → optimize_3d_2d.py → solve_to_bundle.py. 머리카락: capture_hair_solve_prior.py → fit_hair_solve.py → accept_hair_solve.py → probe_hair_only.py. 캡처 명령은 heavy runner에서 실행한다.','',
            f"기본 격자 정규화: {solve['regularization']}. 실제 포즈별 계수는 hair-solve.json의 regularization에 있다. 중립 3포즈와 restY435 이상은 보정 0이다. 3D 사전·파츠·텍스처·메시 토폴로지는 변경하지 않았다.",'',
            '현재 수용 기준은 실제 렌더 오차의 개선 또는 동등, 중립 불변, 2D 차트 무접힘과 잔차 예산이다. 과거 cap과 절대 원화 오차는 진단값으로 보존하며 위 남은 원화 오차가 모두 해소됐다는 뜻은 아니다.','']
    (out / 'hair-low-dof-report.md').write_text('\n'.join(lines))
    before_after_sheet(root,out/'captures',data['poses'])
    paths = [root / 'bundle/index.html', *[out / 'captures' / name for name in ['faceless_hair_9dir_comparison.png', 'hair_only_9dir_overlay.png', 'viewer_9dir_grid.png','r15_before_after_9dir.png']]]
    (out / 'hair-low-dof-sha256.txt').write_text(''.join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n" for path in paths))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    write_release_report(root, 'docs', json.loads((root / 'docs/hair-r15-metrics.json').read_text()))
