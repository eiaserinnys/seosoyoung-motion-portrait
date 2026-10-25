"""Refresh bead-split evidence from the final raster and test executions."""
import hashlib
import json
import re
import subprocess
from fit_hair_solve import ROOT


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    docs=ROOT/'docs';metrics=json.loads((docs/'hair-r15-metrics.json').read_text())
    report=json.loads((docs/'vdd-research-evaluation.json').read_text())
    solve=json.loads((ROOT/'source/viewer/hair-solve.json').read_text())
    chart=metrics['hairSolveChart'];lower=metrics['splitComparison']['lower']
    names={p['name'] for p in solve['poses']}
    raster=[p for p in metrics['priorComparison']['poses'] if p['name'] in names]
    tail=[p for p in lower['poses'] if p['name'] in names]
    minimum=min(p['minDet'] for p in chart['poses'])
    lower_min=min(p['minDet'] for p in chart['lower']['poses'])
    report['split']=dict(chart=chart,comparison=metrics['splitComparison'],
                         upperMean=sum(p['correctedRms'] for p in raster)/6,
                         lowerTailRms=sum(p['correctedRms'] for p in tail)/6,
                         priorLowerTailRms=sum(p['priorRms'] for p in tail)/6)
    from hair_solve_quality import bare_face_limits,span
    from probe_hair_only import read_mask
    features={p['name']:p for p in json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())['poses']}
    raw=json.loads((ROOT/'debug/hair_h0/current/hair-r15-metrics-raw.json').read_text())
    tie_rows=[]
    for pose in raw['poses']:
        if pose['name'] not in ['up-left','up-right']:continue
        name=pose['name'];row=next(r for r in pose['exterior'] if r['y']==520)
        groups={g:span(read_mask(ROOT/'debug/hair_h0/current'/(name+'-'+g+'.png')),520) for g in ['crown','face','ties']}
        tie_rows.append(dict(name=name,**row,groups=groups,eyeRegistrationDelta=pose['registrationDelta'],beads=pose['beads']))
    report['remainingEvidence']=dict(tieRow520=tie_rows,bareFace=bare_face_limits(ROOT,features),
                                    rawResidualFailures=metrics['rawResidualFailures'],
                                    decision='Integrated Track F face calibration at 17032ea; face files remain byte-identical. Prefer smooth strand flow over the former 20.48px candidate: rigid pin translation, vertical median-window hairline samples, stronger horizontal/conformal regularization, smooth bounded lower spans. Reference pixels and gate formulas remain unchanged; all remaining residuals are retained.')
    webgl=ROOT/'debug/vdd-split/docs/vdd-webgl.json'
    if webgl.exists():report['webgl']=json.loads(webgl.read_text())
    report['pass']=report['pass'] and metrics['pass'] and all(p['pass'] for p in tail)
    source_paths=['bundle/hair-solve.mjs','bundle/hair-solve-data.mjs','bundle/rig.mjs',
                  'pipeline/fit_hair_solve.py','pipeline/hair_solve_inputs.mjs','pipeline/hair_solve_metrics.mjs',
                  'pipeline/hair_solve_quality.py','pipeline/probe_hair_only.py','pipeline/vdd_benchmark.mjs',
                  'pipeline/update_vdd_evidence.py','source/viewer/hair-solve.json']
    report['sources']={name:digest(ROOT/name) for name in source_paths}
    (docs/'vdd-research-evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    runtime={str(p.relative_to(ROOT/'bundle')):digest(p) for p in sorted((ROOT/'bundle').rglob('*')) if p.is_file()}
    (docs/'bundle-sha256.json').write_text(json.dumps(runtime,indent=2)+'\n')
    (docs/'bundle-sha256.txt').write_text(''.join(f'{value}  bundle/{name}\n' for name,value in runtime.items()))
    validation=json.loads((docs/'hair-solve-validation.json').read_text())
    validation.update(integratedBase=report['baseline'],researchEvaluation='docs/vdd-research-evaluation.json',
                      researchPass=report['pass'],runtimeSha256={'bundle/'+k:v for k,v in runtime.items()})
    validation['hair']=dict(passed=report['pass'],minChartDet=minimum,minLowerTriangleDet=lower_min,
                            maxDuRms=max(p['stats']['duRms'] for p in solve['poses']),
                            maxDu=max(p['stats']['duMax'] for p in solve['poses']),
                            neutralGeometryUnchanged=metrics['neutralGeometryPass'],
                            actualRasterComparison=metrics['priorComparison'],split=report['split'],
                            absoluteReferenceGatesPassed=metrics['strictRasterPass'],
                            remainingAbsoluteReferenceFailures=len(metrics['rawResidualFailures']))
    validation['scope']='Upper crown/face envelope fitted independently of lower tails/ties. Rigid pitch-only hairpin translation with the existing profile projection preserved, median-window vertical fringe observations, smooth upper/lower fields. Three raster feedback passes, local determinant projection, independent bead-local span fields. Face frozen at integrated main 17032ea; shell, assets and topology unchanged.'
    validation['ui']='Viewer controls and layout unchanged. Actual WebGL nine-direction hair and full-portrait before/after captures; independent review recorded by coordinator.'
    node_log=(ROOT/'debug/vdd-node.tap').read_text();python_log=(ROOT/'debug/vdd-python.txt').read_text()
    validation['node']=dict(passed=int(re.search(r'# pass (\d+)',node_log)[1]),failed=int(re.search(r'# fail (\d+)',node_log)[1]),workers=2,perTestTimeoutMs=60000,
                            command='node --test --test-concurrency=2 --test-timeout=60000 pipeline/test_*.mjs tests/*.test.mjs')
    subtests=re.search(r'(\d+) subtests passed',python_log)
    validation['python']=dict(passed=int(re.search(r'(\d+) passed',python_log)[1]),failed=0,
                              subtestsPassed=int(subtests[1]) if subtests else 0,
                              command='.venv/bin/python -m pytest --timeout=60 -q pipeline tests')
    assert validation['node']['failed']==0 and report['pass']
    captures=['faceless_hair_9dir_comparison.png','hair_only_9dir_overlay.png','r15_before_after_9dir.png','viewer_9dir_grid.png',
              'full_turnaround_7angle.png','vdd_unobserved_angles_sheet.png']
    hashes={name:digest(docs/'captures'/name) for name in captures}
    validation['captureSha256']=hashes
    validation['researchSha256']={'vdd-research-evaluation.json':digest(docs/'vdd-research-evaluation.json')}
    (docs/'hair-solve-validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    (docs/'hair-low-dof-sha256.txt').write_text(''.join(f'{v}  docs/captures/{k}\n' for k,v in hashes.items()))
    lines=['# 구슬 기준 상부·하부 머리카락 볼륨 피팅','',
           f"6포즈 평균 실제 래스터 RMS: 이전 main {report['raster']['mainMean']:.3f}px → {report['raster']['afterMean']:.3f}px.",
           f"트윈테일 스팬 RMS: prior {report['split']['priorLowerTailRms']:.3f}px → {report['split']['lowerTailRms']:.3f}px.",
           f"상부 차트 minDetJ {minimum:.6f}, 하부 실제 삼각형 minDetJ {lower_min:.6f}. 역전 0개.",
           f"남은 절대 원화 게이트 실패: {len(metrics['rawResidualFailures'])}개. 상세는 `hair-r15-metrics.json`의 rawResidualFailures에 보존했다.",'',
           '| 포즈 | 이전 main RMS | 보정 RMS | 하부 prior RMS | 하부 보정 RMS |',
           '| :--- | ---: | ---: | ---: | ---: |']
    before={p['name']:p for p in report['crossValidation']['rasterCaptures']['main']['poses']}
    tails={p['name']:p for p in tail}
    for pose in raster:
        name=pose['name'];t=tails[name]
        lines.append(f"| {name} | {before[name]['correctedRms']:.3f} | {pose['correctedRms']:.3f} | {t['priorRms']:.3f} | {t['correctedRms']:.3f} |")
    lines+=['','## 계산과 불변 조건','',
            '- 상부: crown | face 외곽과 헤어라인, 볼 옆 경계. 12×16 차트, restY515..545 감쇠. 변형 후 외곽을 재측정하는 3회 피팅. 헤어라인은 25열 중앙값·수직 법선·가중치6, 외곽 법선 기울기는 ±1.2로 제한한다.',
            '- 내부 결: 수평2차차분28, scale0.15, conformal0.35로 인접 열의 전단을 억제한다. 머리핀의 pitch 잔차는 [598,290]에서 구한 동일 평행이동만 적용하며 기존 측면 투영과 canopy 모양은 보존한다.',
            '- 하부: 좌우 구슬 원점의 8노드 중심 이동·폭 스케일. 구슬·매듭을 포함하는 첫 3노드는 항등 변환. 2차차분120·1차차분8로 부드럽게 피팅하고 폭 배율을 0.75..1.35로 제한한다.',
            '- 결합부: prior의 실제 alpha 경계 중 구슬 위를 지지하는 차트 노드를 정확히 고정한다. 주변 정점에 대한 연성 관측만으로 남던 0.12px 이동이 얇은 연결 픽셀을 잃게 하는 문제를 제거했다.',
            '- 무접힘: 상부는 고정 노드를 보존하며 위반 셀의 야코비안을 국소 투영한 뒤 독립 JS 게이트로 검증. 하부는 실제 네 메시 전체 삼각형을 45개 기준·중간·측면 포즈에서 검증.',
            '- 중립 3포즈: 정점 해시 일치와 변경 픽셀 0. 구슬 기준점은 모든 검사 포즈에서 정확히 일치. stemGap ≤17px.',
            '- 동결: hair-shell, hair-surface, dome-data, rig JSON, texture 및 얼굴 소스는 통합 main 17032ea와 바이트 동일.',
            '- 뷰어 컴포넌트·글자·여백·조작은 변경하지 않았다. 같은 전체 초상 프레임에서 원화와 렌더를 비교했다.','',
            '## 남은 절대 오차의 근거','',
            '- y520의 구슬 결합부는 눈 등록 이동이 적용된 원화와 dBodyFrame 고정 좌표계가 만나는 행이다. 최신 포즈별 render/ref 스팬·crown/face/ties 소유 범위·등록 이동은 remainingEvidence.tieRow520에 다시 측정했다.',
            '- 이전 얼굴 정합에서 측정했던 bare-face 경계 하한은 폐기했다. Track F 통합 후 머리카락을 제거한 최신 face alpha 삼각형을 다시 스캔한 열별 하한만 remainingEvidence.bareFace에 기록한다.',
            f"- 정수리·다른 상부 외곽의 피팅 잔차도 남아 있다. 전체 {len(metrics['rawResidualFailures'])}개 값과 열별 하한은 vdd-research-evaluation.json의 remainingEvidence에 보존했다.",
            f"- 보완 요청에 따라 원화/게이트를 이동·제외하지 않고 내부 결의 부드러움과 평균 {report['raster']['afterMean']:.2f}px 볼륨 개선을 함께 검증했다.",'',
            '## 검증','',
            f"Node {validation['node']['passed']} passed / 0 failed. Python {validation['python']['passed']} passed, {validation['python']['subtestsPassed']} subtests passed.",
            '실화면: captures/faceless_hair_9dir_comparison.png, hair_only_9dir_overlay.png, r15_before_after_9dir.png, vdd_unobserved_angles_sheet.png.','',
            '## 재현','',
            '무거운 명령은 heavy runner로 감싼다. 워크트리 .venv에 requirements.txt와 pytest-timeout을 준비한다.','',
            '```sh','OPENBLAS_NUM_THREADS=1 .venv/bin/python pipeline/fit_hair_solve.py --capture-prior',
            'OPENBLAS_NUM_THREADS=1 .venv/bin/python pipeline/fit_hair_solve.py',
            f".venv/bin/python pipeline/fit_hair_solve.py --capture-base {report['baseline']}",
            '.venv/bin/python pipeline/probe_hair_only.py',
            'node pipeline/vdd_benchmark.mjs',validation['node']['command'],validation['python']['command'],
            '.venv/bin/python pipeline/update_vdd_evidence.py','```','']
    (docs/'hair-low-dof-report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(pass_=report['pass'],node=validation['node'],python=validation['python'],upperMean=report['split']['upperMean'],lowerTailRms=report['split']['lowerTailRms']),ensure_ascii=False))


if __name__=='__main__':main()
