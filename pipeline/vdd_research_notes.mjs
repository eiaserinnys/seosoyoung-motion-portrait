// Project adaptations are identified separately from each paper's own method.
export const research={
 papers:[
  {title:'View-Dependent Deformation for 2.5-D Cartoon Models',authors:'Fukusato & Maejima',year:2022,url:'https://doi.org/10.1109/MCG.2022.3174202',preprint:'https://arxiv.org/abs/2103.15472',idea:'Blend authored view-dependent 2D shapes',adaptation:'Reuse the existing trigonometric cardinal basis through yaw=0; this basis is a project choice, not a reproduction of the paper'},
  {title:'3D Character Reconstruction from Hand-drawn Model Sheets',authors:'Yoon et al. (POSTECH)',year:2026,url:'https://doi.org/10.1111/cgf.70323',code:'https://github.com/hjyoon02/ModelSheetTo3D',idea:'Separate shared 3D geometry from per-view screen correspondence corrections',adaptation:'Keep authored 3D geometry and constrain the residual chart with discrete CR energy. This linear conformal regularizer is a project adaptation, not the full mesh/Jacobian reconstruction pipeline',codeReuse:'No code copied; official repository root has no license file in the inspected main tree'},
  {title:'Inverse Rig Optimization from Line Drawings',year:2026,url:'https://arxiv.org/html/2609.00732v1',idea:'Screen-space contour-normal loss and controlled rig deviation (§3.5)',adaptation:'Retain the existing normal-distance loss, attenuate lower/steep silhouette samples as a visibility confidence heuristic; y/slope weights are project-specific, not learned occlusion or surface-normal estimation'},
  {title:'View-Dependent Deformation Fields for 2D Editing of 3D Models',authors:'El Mqirmi & Aigerman',year:2025,url:'https://arxiv.org/html/2504.05544v1',idea:'Smooth, view-dependent 2D corrections lifted/composed over existing geometry',adaptation:'Smooth neutral-yaw interpolation and independent 60..88 degree support fade; keep the existing additive residual rather than introducing compositional 3D deformation layers'}
 ],
 applied:[
  'C1 trigonometric yaw weights through zero; cardinal nine keys unchanged; smoothstep fade from 60 to 88 degrees isolates profile parts',
  'Silhouette confidence: 1.25 at y<=220, 0.55 at 260..360, otherwise 0.85; divide by 1+0.5*slope^2',
  'Area-weighted cell-center Cauchy-Riemann energy on (160*s,restY) pixels (lambda=0.5); mirror coupling between diagonal charts (lambda=0.2); weak adjacent-yaw chart prior (lambda=0.05) also predicts neutral LOPO',
  'Separate pin, held-out normal-observation prediction, raster error, sampled Jacobians, neutral-yaw derivatives and actual WebGL sweep'
 ],
 deferred:[
  {item:'POSTECH full 3D mesh reconstruction and ray deformation pipeline',reason:'Would replace the approved authored 3D prior, topology and texture workflow; residual chart improvements meet the present scope'},
  {item:'Inverse Rig neural MLP surrogate',reason:'The existing linear normal-equation solve is directly differentiable and small; training a surrogate adds approximation error and dependencies'},
  {item:'Fixed contour vertex IDs at +/-90 degrees',reason:'Visible silhouette identities change with occlusion and dedicated profile parts; preserve the existing measured profile path'},
  {item:'Full nonlinear ARAP',reason:'Discrete CR penalizes shear/anisotropic scale with a single linear solve; it permits uniform scale and rotation, so it is documented as conformal rather than equated to ARAP'}
 ],
 limitations:[
  'Twelve unobserved angle combinations have no independent reference drawings; their metrics measure orientation, conformal distortion and mirror consistency, not held-out image accuracy',
  'LOPO evaluates normal-distance observations on a fixed prior-visible domain; it is separate from the actual raster acceptance metric',
  'The unchanged shared 3D prior and face fit were calibrated from all original drawings. LOPO here withholds residual-chart observations, not every upstream drawing-derived asset',
  'C1 is guaranteed through yaw=0; the existing clamp at +/-45 remains. Support fading is C1 at 60 and 88',
  'Conformal energy at intermediate views can increase while raster error and symmetry improve; report measured before/after values without using this diagnostic as a fitted success claim'
 ],
 selection:{discarded:'Uniform regularizer scale per pitch triplet: raster mean 37.359px exceeded main 37.281px',adopted:'Pose-specific budget multipliers with the same conformal, symmetry and view coefficients; no raster-driven hyperparameter grid search',units:'du in pixels; chart x=160*s; CR rows area weighted; symmetry/view sum squared chart-node displacement'},
 reproduction:[
  'python3 pipeline/capture_vdd_baselines.py',
  'OPENBLAS_NUM_THREADS=1 python3 pipeline/fit_hair_solve.py',
  'python3 pipeline/accept_hair_solve.py',
  'node pipeline/vdd_benchmark.mjs',
  'python3 pipeline/capture_vdd.py',
  'python3 pipeline/probe_hair_only.py',
  'python3 pipeline/update_vdd_evidence.py'
 ]
};
