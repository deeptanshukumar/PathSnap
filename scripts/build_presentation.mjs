// Optional deck authoring: uses Codex's Artifact Tool 2.8.134 runtime.
// Model training, inference and the static Mac UI do not require Node.js.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const workspace = process.cwd();
const build = path.join(workspace, '.build/presentation');
const output = path.join(workspace, process.env.PRESENTATION_OUTPUT || 'reports/PathSnap_Presentation_Final.pptx');
await fs.mkdir(build, {recursive:true});
let artifact;
if (process.env.RUNTIME_NODE_MODULES) {
  const requireRuntime = createRequire(path.join(process.env.RUNTIME_NODE_MODULES, '__runtime__.cjs'));
  artifact = await import(pathToFileURL(requireRuntime.resolve('@oai/artifact-tool')).href);
} else artifact = await import('@oai/artifact-tool');
const {Presentation, PresentationFile} = artifact;
const skill = process.env.ARTIFACT_SKILL_DIR;
if (!skill) throw new Error('ARTIFACT_SKILL_DIR must point to the presentation skill for validation');
const {finalizePresentation, resolvePresentationFont, applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font = resolvePresentationFont({fontFamily:'Arial'});
const read = async name => JSON.parse(await fs.readFile(path.join(workspace,'artifacts',name),'utf8'));
const models = await read('model_comparison.json');
const tuning = await read('tuning.json');
const trials = await read('bayesian_trials.json');
const decoded = await read('decoder_comparison.json');
const manifest = await read('training_manifest.json');
const final = await read('final_model.json');
const n=decoded.reduce((sum,r)=>sum+r.observations,0);
const pooled=(key,metric)=>decoded.reduce((sum,r)=>sum+r.observations*r[key][metric],0)/n;
const mean=(key,metric)=>decoded.reduce((sum,r)=>sum+r[key][metric],0)/decoded.length;
const C={ink:'#163d35',green:'#087f6a',muted:'#63756b',gold:'#c59a42',blue:'#5c7ea0',white:'#ffffff',light:'#f2f6f1'};
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
function text(slide,value,x,y,w,h,size=24,color=C.ink,bold=false){
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;shape.text.style={typeface:font,fontSize:size,color,bold,wrap:'square',autoFit:'none',insets:0};return shape;
}
function slide(title,notes='',dark=false){
  const s=presentation.slides.add();s.background.fill=dark?C.ink:C.white;
  text(s,title,64,46,1152,76,40,dark?C.white:C.ink,true);
  text(s,'PT-088 · PathSnap · 18 November 2026',64,680,800,22,14,dark?'#bed4c8':C.muted);
  text(s,String(presentation.slides.items.length),1170,680,46,22,14,dark?'#bed4c8':C.muted);
  s.speakerNotes.text=notes;return s;
}
function table(slide,values,x,y,w,h,widths){
  const t=slide.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
  t.borders.assign({fill:'#d9e3da',width:0.6,style:'solid'});
  for(let i=0;i<values.length;i++)for(let j=0;j<values[i].length;j++){
    const cell=t.getCell(i,j);cell.fill=i===0?C.ink:(i%2===0?C.light:C.white);
    cell.text.style={typeface:font,fontSize:19,color:i===0?C.white:C.ink,bold:i===0,insets:12};
  }return t;
}
function chart(slide,type,categories,series,position,options={}){
  // Office literal chart workbooks preserve 15 significant digits. Six decimal
  // places retain more precision than displayed; full results stay in JSON.
  series=series.map(s=>({...s,smooth:false,values:s.values.map(v=>Number(v.toFixed(6)))}));
  const c=slide.charts.add(type,{position,categories,series,hasLegend:series.length>1,
    legend:{position:'bottom',overlay:false,textStyle:{fontSize:17,fill:C.muted}},
    chartFill:C.white,plotAreaFill:C.white,
    xAxis:{textStyle:{fontSize:17,fill:C.muted},majorGridlines:null},
    yAxis:{min:0,max:1,numberFormatCode:'0%',textStyle:{fontSize:16,fill:C.muted},majorGridlines:{fill:'#e4eae3',width:1}},
    ...options});applyPresentationChartFont(c,{fontFamily:font});return c;
}
async function picture(slide,filename,x,y,w,h){
  const b=await fs.readFile(path.join(workspace,'reports/evidence',filename));
  slide.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/jpeg',alt:filename,
                    fit:'contain',position:{left:x,top:y,width:w,height:h}});
}

{
const s=slide('PathSnap','Opening: Team PT-088, four members. Explain the GPS-to-road reconstruction problem. This is an independently implemented required baseline, not a novelty claim. Map source: actual student_test trip 00000000 and our saved model; reports/maps/development_00000000.html. AI assistance is disclosed in the report.',true);
text(s,'Distributed map matching\nand route reconstruction',64,156,480,150,37,C.white);
text(s,'Five measured models\nBayesian optimisation\nDirected Viterbi decoding',64,357,450,130,25,'#bed4c8');
text(s,'Chiranth · Darshan · Bhuvan · Deeptanshu',64,576,480,40,22,'#bed4c8');
await picture(s,'folium_development.jpg',570,150,646,416);
text(s,'Actual development trip · reference in gold · prediction in green',570,579,645,48,17,'#bed4c8');
}
{
const s=slide('Dataset and evaluation boundaries','Sources: five organiser PDFs and artifacts/inventory.json. Mention the reference route is not a per-GPS label file. Do not describe development accuracy as final evaluator accuracy.');
table(s,[['Partition','Trips','GPS observations','Use'],['Training','80','208,417','Fit + grouped CV'],['Student test','10','19,160','Development comparison'],['Evaluator test','10','19,674','Frozen final evaluation']],64,150,1152,250,[260,130,260,502]);
text(s,'Four files per trip',64,444,530,40,29,C.green,true);
text(s,'.track: GPS + seconds\n.nodes / .arcs: directed network\n.route: ordered reference arc sequence',64,498,550,130,24);
text(s,'Real data quality issues',690,444,526,40,29,C.green,true);
text(s,'153,064 s maximum training gap\nDisconnected road components\nReference endpoint breaks',690,498,526,130,24);
}
{
const s=slide('Candidate features and proxy supervision','Sources: pathsnap/features.py, pathsnap/reference.py, artifacts/training_manifest.json and inventory.json. Every fifth training observation is selected after a fixed 75 m label-quality filter. All 19,160 development observations are scored. The exclusion count is before stride selection. Local equirectangular geometry is an approximation, up to a network dimension near 77.6 km.');
text(s,'Up to 8 directed candidates within 100 m',64,150,1120,48,30,C.green,true);
text(s,'13 route-independent features\n\nDistance and relative distance\nHeading, speed and time gaps\nEdge geometry and junction degrees',64,225,560,220,25);
chart(s,'bar',['Retained','Excluded >75 m'],[{name:'Training GPS observations before stride',values:[187027,21390],fill:C.green}],{left:690,top:225,width:515,height:300},{hasLegend:false,yAxis:{min:0,max:210000,numberFormatCode:'#,##0',textStyle:{fontSize:16,fill:C.muted}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'#,##0',textStyle:{fontSize:19,fill:C.ink}}});
text(s,'Monotone alignment creates proxy pointwise labels',64,538,1120,42,29,C.green,true);
text(s,'Route order constrains the assignment. It does not establish physical per-point ground truth.\nCandidate generation and prediction never read the reference route.',64,592,1152,64,22);
}
{
const s=slide('Five models on identical trip folds','Sources: artifacts/model_comparison.json and folds.json. Three trip-grouped folds, seed 357. CV candidate Macro F1 selects the classifier family. Student-test metrics remain development estimates. Explain that candidate Macro F1 is a binary classification metric and Top-1 is within-observation ranking accuracy against proxy labels.');
const labels=['Logistic','Gaussian NB','Decision tree','Random forest','Hist. boosting'];
chart(s,'bar',labels,[{name:'CV candidate Macro F1',values:models.map(r=>r.cv_candidate_macro_f1),fill:C.green},{name:'Development proxy Top-1',values:models.map(r=>r.student_test.summary.proxy_top1_accuracy),fill:C.blue}],{left:64,top:142,width:1152,height:360},{barOptions:{direction:'column',grouping:'clustered',gapWidth:85},yAxis:{min:0.6,max:1,numberFormatCode:'0%',textStyle:{fontSize:16,fill:C.muted}}});
text(s,'Histogram gradient boosting leads CV: '+models.at(-1).cv_candidate_macro_f1.toFixed(4),64,537,1152,46,30,C.green,true);
text(s,'Training and validation trips never overlap. Scalers fit inside each fold.\nFull configurations, fit times, precision, recall, log loss and confusion matrices are in the report.',64,598,1152,60,22);
}
{
const s=slide('Bayesian optimisation','Sources: artifacts/bayesian_trials.json and tuning.json. Optuna TPE seed 357, four random startup trials and eight adaptive trials. Objective is mean trip-grouped CV binary candidate Macro F1, not development accuracy. The tuned model improves CV but slightly lowers development Top-1; say this explicitly. Official reference: https://optuna.readthedocs.io/en/v4.6.0/reference/samplers/generated/optuna.samplers.TPESampler.html');
const v=trials.map(t=>t.value);let best=-Infinity;
chart(s,'line',trials.map(t=>String(t.number)),[{name:'Trial objective',values:v,line:{fill:C.blue,width:2},marker:{symbol:'circle',size:6}},{name:'Best so far',values:v.map(x=>best=Math.max(best,x)),line:{fill:C.green,width:3}}],{left:64,top:150,width:740,height:365},{yAxis:{min:0.925,max:0.930,numberFormatCode:'0.000',textStyle:{fontSize:16,fill:C.muted}},xAxis:{title:'Trial index',textStyle:{fontSize:16,fill:C.muted}}});
text(s,'12 completed trials',855,168,360,50,31,C.green,true);
text(s,'4 startup + 8 adaptive\nSame 3 training-trip folds\nNo evaluator data',855,240,360,130,24);
text(s,'CV Macro F1\n'+tuning.untuned_cv_macro_f1.toFixed(4)+' → '+tuning.cv_candidate_macro_f1.toFixed(4),855,414,360,120,31,C.green,true);
text(s,'Best: learning rate '+tuning.best_parameters.learning_rate.toFixed(4)+', '+tuning.best_parameters.max_leaf_nodes+' leaves, L2 '+tuning.best_parameters.l2_regularization.toFixed(3)+', minimum leaf '+tuning.best_parameters.min_samples_leaf,64,567,1152,58,23);
text(s,'The tuned model improves CV; development Top-1 changes '+(100*tuning.untuned_student_test.summary.proxy_top1_accuracy).toFixed(2)+'% → '+(100*tuning.tuned_student_test.summary.proxy_top1_accuracy).toFixed(2)+'%',64,628,1152,38,21,C.muted);
}
{
const s=slide('Directed HMM-style decoding','Sources: pathsnap/decoder.py, geometry.py, artifacts/decoder_search.json. Newson & Krumm (2009): https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/. Discriminative emissions and compatibility potentials mean HMM-style sequence decoding, not a newly trained generative HMM. The beta and normalisation settings use permitted student-test development data.');
text(s,'State: directed arc + projected position',64,146,1152,42,30,C.green,true);
text(s,'Emission: normalised classifier candidate score\nTransition: directed shortest-path distance versus GPS movement',64,212,1152,86,26);
text(s,'δ(t,j) = log e(t,j) + maxᵢ [ δ(t−1,i) + log T(i,j) ]',64,335,1152,75,35,C.ink,true);
table(s,[['Condition','Decoder action'],['GPS gap >60 seconds','Start an independent route piece'],['No legal directed transition','Reset and report the break'],['Candidate radius miss','Flag a nearest-sample fallback'],['Reachable consecutive states','Insert actual shortest-path bridge arcs']],64,437,1152,208,[445,707]);
}
{
const s=slide('Route coherence improves; pointwise accuracy falls','Sources: artifacts/decoder_comparison.json. Route overlap and continuity are means over ten trips; pointwise accuracy pools all 19,160 observations. HMM bridged continuity includes inserted intermediate arcs. Classifier continuity has no bridge stage, so some improvement is due to reconstruction, not Viterbi alone. HMM-state direct continuity is '+mean('hmm','direct_arc_continuity').toFixed(4)+'. Overlap uses unique directed arcs weighted by length; LCS adds sequence ordering.');
chart(s,'bar',['Proxy pointwise accuracy','Route overlap F1','Direct arc continuity'],[{name:'Classifier only',values:[pooled('classifier','proxy_top1_accuracy'),mean('classifier','route_length_f1'),mean('classifier','direct_arc_continuity')],fill:C.blue},{name:'Viterbi + reconstructed route',values:[pooled('hmm','proxy_top1_accuracy'),mean('bridged_route','route_length_f1'),mean('bridged_route','direct_arc_continuity')],fill:C.green}],{left:64,top:142,width:1152,height:400},{barOptions:{direction:'column',grouping:'clustered',gapWidth:100},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0%',textStyle:{fontSize:19,fill:C.ink}}});
text(s,'Overlap: 0.934 → 0.957   |   Ordered route LCS F1: 0.733 → 0.801',64,576,1152,45,28,C.green,true);
text(s,'The ten development trips select decoder settings and measure these results.\nPointwise scores use inferred reference labels; reserved results appear on the final slide.',64,627,1152,55,21,C.muted);
}
{
const s=slide('Decoder gains vary across trips','Source: artifacts/decoder_comparison.json. Each trip has its own road network and reference. Show both improvements and regressions. Do not claim a cause from this plot alone. Trip 00000077 remains a prominent failure case; 00000052 is easy but must not be used as the only demonstration.');
chart(s,'bar',decoded.map(r=>r.trip.slice(-2)),[{name:'Classifier proxy Top-1',values:decoded.map(r=>r.classifier.proxy_top1_accuracy),fill:C.blue},{name:'Viterbi proxy Top-1',values:decoded.map(r=>r.hmm.proxy_top1_accuracy),fill:C.green}],{left:64,top:150,width:1152,height:380},{barOptions:{direction:'column',grouping:'clustered',gapWidth:80},xAxis:{title:'Trip ID (last two digits)',textStyle:{fontSize:18,fill:C.muted}}});
text(s,'Improvement on 5 trips; regression on 5 trips',64,568,1152,43,30,C.green,true);
text(s,'Network structure, reference uncertainty and local candidate ambiguity create a real trade-off.\nThe system returns explicit resets instead of drawing invented links across outages.',64,625,1152,58,22);
}
{
const s=slide('API and evaluator interface','Sources: actual browser screenshot reports/evidence/ui_working.jpg and real HTTP results reports/evidence/api_smoke.json. This screenshot shows static UI at port 8010 calling API port 8000 on the same physical host. It proves cross-origin local integration, not Mac-to-Linux operation. All ten live HTTP requests succeeded. Reference geometry is omitted from all API maps.');
await picture(s,'ui_working.jpg',64,133,730,514);
text(s,'Linux laptop',850,161,365,42,30,C.green,true);
text(s,'FastAPI + trained model\nInference, decoding and scoring\nReference data stays here',850,216,365,120,23);
text(s,'Chiranth’s Mac',850,375,365,42,30,C.green,true);
text(s,'Static UI on port 8010\nAPI URL = Linux LAN IP:8000\nTen trips loaded from API',850,430,365,120,23);
text(s,'20/20 HTTP trips passed\nPhysical two-laptop test pending',850,587,365,66,22,C.muted);
}
{
const s=slide('Demonstration and remaining checks','Sources: README.md, docs/PROJECT_STATE.md, actual tests and smoke evidence. The reserved archive arrived after the predictor freeze and has now been evaluated. GNOME denied native terminal-window capture; the team must take a genuine screenshot during the two-laptop demo. Bhuvan has not yet supplied a novelty, so do not claim one. AI disclosure: Codex assisted implementation, experiments and drafting; the team must understand and review the claims.');
table(s,[['Ready','To complete with the team'],['Five models + 12 TPE trials','Connect Chiranth’s Mac to this API'],['Viterbi and Folium maps','Capture receiving-terminal screenshot'],['12 passing correctness/API tests','Rehearse selection of reserved trips'],['Live HTTP for every development trip','Review report and rehearse explanations']],64,150,1152,280,[576,576]);
text(s,'Demo sequence',64,482,1152,42,30,C.green,true);
text(s,'Start API → connect Mac UI → select any trip → reconstruct\nCompare the request ID with the receiving server’s log',64,542,1152,80,26);
text(s,'Baseline methods are established. Bhuvan’s novelty remains open. AI assistance is disclosed.',64,634,1152,35,20,C.muted);
}


{
const result=await read('evaluator_results.json');const rows=result.requests;
const total=rows.reduce((sum,r)=>sum+r.observations,0);
const acc=k=>rows.reduce((sum,r)=>sum+r.observations*r.metrics[k].proxy_top1_accuracy,0)/total;
const avg=(k,m)=>rows.reduce((sum,r)=>sum+r.metrics[k][m],0)/rows.length;
const s=slide('Reserved evaluation after model freeze','Sources: artifacts/evaluation_freeze.json, evaluator_manifest.json and evaluator_results.json/.csv. Freeze commit a61ce50 predates archive access. All ten trips completed actual HTTP inference. No evaluator result changed any model, feature or decoder setting. Pointwise accuracy uses the frozen proxy alignment convention. These results do not establish a universal HMM improvement; sequence F1 after bridging also decreases.');
table(s,[['Reserved metric','Classifier only','Viterbi / bridge'],['Proxy pointwise accuracy',(100*acc('classifier')).toFixed(2)+'%',(100*acc('hmm')).toFixed(2)+'%'],['Mean route overlap F1',avg('classifier','route_length_f1').toFixed(4),avg('bridged_route','route_length_f1').toFixed(4)],['Mean direct continuity',avg('classifier','direct_arc_continuity').toFixed(4),avg('bridged_route','direct_arc_continuity').toFixed(4)],['Mean ordered route LCS F1',avg('classifier','ordered_route_lcs_f1').toFixed(4),avg('bridged_route','ordered_route_lcs_f1').toFixed(4)]],64,150,1152,300,[552,300,300]);
text(s,'10 reserved trips · 19,674 observations · 10 successful HTTP responses',64,498,1152,50,29,C.green,true);
text(s,'Road overlap and continuity improve, while proxy pointwise accuracy and bridged sequence F1 fall.\nThese are final measurements; the predictor remains frozen.',64,571,1152,88,24);
}

const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidate);
for(const [i,s] of presentation.slides.items.entries()){
  const png=await presentation.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const result=await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath:output,
  pythonExecutable:process.env.RUNTIME_PYTHON||'python3',
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
              '--require-native-table-slide','2','--require-native-table-slide','6','--require-native-table-slide','10','--require-native-table-slide','11'],
  requiredNativeTableOwnerSlides:[2,6,10,11],requiredNativeChartOwnerSlides:[3,4,5,7,8],
  fontPolicy:{basis:'design',families:[font]},materializeLiteralChartWorkbooks:true,
  verifyArtifactToolImport:true,receiptPath:path.join(build,path.basename(output)+'.validation.json')});
console.log(JSON.stringify(result));
