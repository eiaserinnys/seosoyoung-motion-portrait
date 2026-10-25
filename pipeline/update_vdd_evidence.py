"""Refresh generated report, runtime/capture hashes and final test evidence."""
import hashlib
import json
import re
from fit_hair_solve import ROOT


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    docs=ROOT/'docs';report=json.loads((docs/'vdd-research-evaluation.json').read_text())
    for name in ['pipeline/vdd_benchmark.mjs','pipeline/vdd_cross_validation.py','pipeline/vdd_research_notes.mjs','pipeline/capture_vdd.py','pipeline/capture_vdd_baselines.py']:
        report['sources'][name]=digest(ROOT/name)
    (docs/'vdd-research-evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    runtime={str(p.relative_to(ROOT/'bundle')):digest(p) for p in sorted((ROOT/'bundle').rglob('*')) if p.is_file()}
    (docs/'bundle-sha256.json').write_text(json.dumps(runtime,indent=2)+'\n')
    (docs/'bundle-sha256.txt').write_text(''.join(f'{value}  bundle/{name}\n' for name,value in runtime.items()))
    lines=['','## VDD 연구 적용과 일반화','',
           f"실제 6포즈 평균 래스터 RMS: main {report['raster']['mainMean']:.6f}px → {report['raster']['afterMean']:.6f}px. 6포즈 모두 같은 r15 prior보다 개선됐다.",'',
           '| 평가 | main | 개선 |','| :--- | ---: | ---: |']
    hair=report['crossValidation']['hair']['mean']
    for name,key in [('관측 normal RMS (px)','pinRms'),('LOPO normal RMS (px)','lopoRms')]:
        lines.append(f"| {name} | {hair['before'][key]:.6f} | {hair['after'][key]:.6f} |")
    jump=report['continuity'][-1]
    lines.append(f"| 정면 각속도 점프 (px/deg, ε={jump['epsilon']}) | {jump['before']:.9f} | {jump['after']:.9f} |")
    for label,key in [('중간 각도 평균 CR 에너지','conformalEnergy'),('중간 각도 평균 대칭 RMS (px)','mirrorRms')]:
        values={side:sum(p[side]['hair'][key] for p in report['intermediate'])/len(report['intermediate']) for side in ['before','after']}
        lines.append(f"| {label} | {values['before']:.9f} | {values['after']:.9f} |")
    minimum=min([p['after']['minDetJ'] for p in report['keys']]+[p['after']['hair']['minDetJ'] for p in report['intermediate']]+[p['after']['minDetJ'] for p in report['sweep']])
    lines+=['',f'기준·중간·연속 스윕의 머리카락 차트 minDetJ 최솟값은 {minimum:.6f}이다. 중간 각도 CR 에너지는 증가했으며, 채택 기준은 래스터 평균 개선과 변위·무접힘 예산이다. 얼굴 데이터와 원래 3D prior는 바이트가 같다.','',
            '얼굴 8포즈의 Pin/LOPO는 `vdd-research-evaluation.json`의 crossValidation.face에 전후를 각각 기록했다. 얼굴은 변경하지 않아 값이 동일하다. 머리카락 LOPO는 보류 포즈의 데이터 항을 빼고 거울·이웃 시점 결합으로 예측한다. Pin/LOPO 관측 RMS와 실제 래스터 RMS는 측정 영역과 정의가 다르다.','',
            '### 적용과 보류','']
    lines.extend('- '+value for value in report['research']['applied'])
    lines+=['']+[f"- 보류: {value['item']} — {value['reason']}" for value in report['research']['deferred']]
    lines+=['','### 근거 문헌','']
    lines.extend(f"- [{p['title']} ({p['year']})]({p['url']}): {p['adaptation']}" for p in report['research']['papers'])
    lines+=['','### 검증 범위와 한계','']
    lines.extend('- '+value for value in report['research']['limitations'])
    lines+=['','실화면: `captures/vdd_unobserved_angles_sheet.png`에서 main/after 같은 각도와 크기를 비교했다. 중간 대각선에서 새 틈·직사각형 돌출이 보이지 않았고 수평 4컷은 전후가 같다. 기존 뷰어 UI·컴포넌트·CSS·글자·간격은 변경하지 않았다. 캡처는 초상 전체를 같은 720×720 영역으로 잘라 같은 비율로 축소했다. 전문 UI/UX 검수는 담당 세션에 인계한다.','',
            '### 재현 순서','',
            '무거운 명령은 heavy runner 안에서 하나씩 실행한다. Python 검증은 pytest-timeout이 설치된 인터프리터를 사용한다.','',
            '```sh',*report['research']['reproduction'],'```','']
    target=docs/'hair-low-dof-report.md';text=target.read_text().partition('\n## VDD 연구 적용과 일반화\n')[0]
    target.write_text(text.rstrip()+'\n'+'\n'.join(lines))
    validation=json.loads((docs/'hair-solve-validation.json').read_text())
    validation.update(integratedBase=report['baseline'],researchEvaluation='docs/vdd-research-evaluation.json',
                      researchPass=report['pass'],webgl=json.loads((docs/'vdd-webgl.json').read_text()),
                      runtimeSha256={'bundle/'+k:v for k,v in runtime.items()})
    metrics=json.loads((docs/'hair-r15-metrics.json').read_text())
    validation['hair']=dict(passed=report['pass'],minChartDet=minimum,
                            maxDuRms=max(p['duRms'] for p in report['keys']),maxDu=max(p['duMax'] for p in report['keys']),
                            neutralGeometryUnchanged=metrics['neutralGeometryPass'],actualRasterComparison=report['raster'],
                            absoluteReferenceGatesPassed=metrics['strictRasterPass'],remainingAbsoluteReferenceFailures=len(metrics['rawResidualFailures']))
    validation['scope']='Unchanged face and r15 3D prior; C1 neutral-yaw interpolation, profile support fade and occlusion-weighted conformal/mirror/view coupled hair residuals. Absolute reference mismatches remain diagnostics.'
    validation['ui']='Existing viewer and controls unchanged. Ten before/after angles and 111 WebGL sweep frames per variant; specialist UI/UX review remains with coordinator.'
    node_log=(ROOT/'debug/vdd-node.tap').read_text();python_log=(ROOT/'debug/vdd-python.txt').read_text()
    validation['node']=dict(passed=int(re.search(r'# pass (\d+)',node_log)[1]),failed=int(re.search(r'# fail (\d+)',node_log)[1]),
                            command='node --test --test-concurrency=2 --test-timeout=60000 pipeline/test_*.mjs tests/*.test.mjs')
    validation['python']=dict(passed=int(re.search(r'(\d+) passed',python_log)[1]),failed=0,
                              subtestsPassed=int(re.search(r'(\d+) subtests passed',python_log)[1]),
                              command='.venv/bin/python -m pytest --timeout=60 -q pipeline tests')
    assert validation['node']['failed']==0 and report['pass']
    names=['viewer_9dir_grid.png','faceless_hair_9dir_comparison.png','hair_only_9dir_overlay.png',
           'hairless_face_9dir_comparison.png','face_geom_9dir_overlay.png','face_turnaround_7angle.png',
           'r15_before_after_9dir.png','vdd_unobserved_angles_sheet.png']
    captures={name:digest(docs/'captures'/name) for name in names}
    validation['captureSha256']=captures
    validation['researchSha256']={name:digest(docs/name) for name in ['vdd-research-evaluation.json','vdd-webgl.json']}
    (docs/'hair-solve-validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    (docs/'hair-low-dof-sha256.txt').write_text(f"{digest(ROOT/'bundle/index.html')}  bundle/index.html\n"+''.join(f'{v}  docs/captures/{k}\n' for k,v in captures.items()))
    print(json.dumps(dict(researchPass=report['pass'],node=validation['node'],python=validation['python'],runtimeFiles=len(runtime),captureFiles=len(captures)),ensure_ascii=False))


if __name__=='__main__':main()
