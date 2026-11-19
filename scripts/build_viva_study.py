"""Generate a separate viva study PDF from actual project results and reviewed source.

This is rehearsal material, not the consolidated submission report. No training,
reserved-data inspection, predictor changes or invented experimental results.
"""
from pathlib import Path
import json
from xml.sax.saxutils import escape
import subprocess

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parent.parent
read = lambda name: json.loads((ROOT / 'artifacts' / name).read_text())
MODELS = read('model_comparison.json')
TUNING = read('tuning.json')
LATER = read('evaluator_model_comparison.json')
DEV = read('decoder_comparison.json')
RESERVED = read('evaluator_results.json')['requests']
NAMES = {'logistic':'Logistic regression','gaussian_nb':'Gaussian Naive Bayes','decision_tree':'Decision tree','random_forest':'Random forest','hist_gradient_boosting':'Histogram gradient boosting'}
SHORT = {'logistic':'Logistic','gaussian_nb':'Gaussian NB','decision_tree':'Decision tree','random_forest':'Random forest','hist_gradient_boosting':'Hist boosting'}
styles = getSampleStyleSheet()
styles.add(ParagraphStyle('BodyStudy',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=6,textColor=colors.HexColor('#203d35')))
styles.add(ParagraphStyle('QuestionStudy',parent=styles['BodyStudy'],fontName='Helvetica-Bold',fontSize=10.5,leading=14,spaceBefore=7,spaceAfter=3))
styles.add(ParagraphStyle('SmallStudy',parent=styles['BodyStudy'],fontSize=8,leading=10.5,spaceAfter=5))
styles.add(ParagraphStyle('TitleStudy',fontName='Helvetica-Bold',fontSize=30,leading=35,spaceAfter=15,textColor=colors.HexColor('#087f6a')))
styles['Heading1'].textColor=colors.HexColor('#087f6a');styles['Heading1'].fontSize=19;styles['Heading1'].leading=24
story=[];ids=[];sections=[]

def p(text,small=False): story.append(Paragraph(text,styles['SmallStudy' if small else 'BodyStudy']))
def heading(text): story.append(Paragraph(text,styles['Heading1']))
def page(title,source=''):
    if story: story.append(PageBreak())
    heading(title);sections.append(title)
    if source:p('Evidence: '+escape(source),True)
def qa(identifier,question,answer):
    assert identifier not in ids,identifier
    ids.append(identifier)
    story.append(KeepTogether([Paragraph(escape(identifier+'  '+question),styles['QuestionStudy']),Paragraph(escape(answer),styles['BodyStudy'])]))
def table(rows,widths=None):
    values=[[Paragraph(escape(str(v)),styles['SmallStudy']) for v in row] for row in rows]
    t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3eee6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f3')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#c1d0c3'))]))
    story.extend([t,Spacer(1,7)])
def pct(x):return f'{100*x:.2f}%'
def avg(rows,part,key):return sum(r[part][key] for r in rows)/len(rows)
def weighted(rows,part,key):return sum(r['observations']*r[part][key] for r in rows)/sum(r['observations'] for r in rows)

story.append(Spacer(1,25));story.append(Paragraph('PathSnap\nViva Study Guide',styles['TitleStudy']))
p('<b>Team PT-088 | 18 November 2026 | Includes Chiranth\'s pushed UI and trace code</b>')
p('Short answers to every explicit question in Deliverables.pdf, plus follow-ups covering all five organiser PDFs and the actual implementation. Read the answer, then explain it once in your own words.')
p('<b>Personal study material:</b> this is separate from the single consolidated technical report required for submission. It is AI-assisted preparation, not a claim that the team has already mastered every explanation.')
qa('START1','What does our system do?','It turns noisy GPS observations into a plausible route on a directed road network. A classifier scores nearby road candidates; Viterbi uses connectivity to choose a sequence; real road bridges fill the spaces between observations.')
qa('START2','What should everyone remember in five minutes?','80 training trips, 10 student-test trips, 10 reserved trips. Five models. Three whole-trip CV folds. Seed 357. Eight candidates and 13 features. Twelve TPE trials: four random startup and eight adaptive.')
qa('START3','What is the most important result caveat?','Our per-GPS labels are inferred from the ordered reference route. Therefore pointwise accuracy is a proxy. More continuous roads do not automatically mean more correct pointwise labels or route order.')
qa('START4','What is still missing for full deliverable compliance?','A genuine receiving-terminal screenshot, verified physical Mac-to-Linux operation, and current-UI evidence in the submission report. The running API also needs restarting to load the newly pushed trace endpoints. See docs/DELIVERABLES_AUDIT.md.')
qa('START5','What is the honest one-sentence result?','The frozen system improves road overlap and continuity, but HMM reduces pooled proxy pointwise accuracy. We keep the failures in the report instead of changing settings after seeing reserved results.')
p('Fast route through this guide: START -> model overview -> your assigned model -> Bayesian search -> HMM/Viterbi -> Chiranth\'s trace pages -> integrity. Then quiz each other using the question IDs.',True)

page('1. Dataset and raw-file questions','Dataset Exploration.pdf Sections 2-9; data.py; inventory.json')
for item in [
('DATA1','What does each file contain?','.track has longitude, latitude and timestamp in seconds. .nodes has longitude and latitude. .arcs has source and target node indices. .route has chronological reference arc IDs.'),
('DATA2','How do rows and IDs relate?','Node IDs are zero-based .nodes row numbers; arc IDs are zero-based .arcs row numbers. A GPS row is an observation, not a road ID. Route row j is the j-th traversed reference arc, not GPS observation j.'),
('DATA3','Are arc IDs global classes?','No. IDs are local to a trip network. Arc 10 on one trip can be unrelated to arc 10 on another. We learn candidate suitability from features, not the numeric ID.'),
('DATA4','What are the partition sizes and roles?','Training: 80 trips and 208,417 observations, for fitting/CV. Student test: 10 trips and 19,160 observations, for development. Reserved: 10 trips and 19,674 observations, for final evaluation after freeze.'),
('DATA5','How is coordinate order handled?','Raw input is longitude then latitude. Folium and Leaflet locations use latitude then longitude, so display code swaps them deliberately.'),
('DATA6','What missing values or time problems did we find?','The 90 training/development trips have no nonfinite GPS values or non-increasing timestamps. Sampling is usually one second, but the largest training gap is 153,064 s and the largest student-test gap is 210 s.'),
('DATA7','What network problems were observed?','Networks can have disconnected components. The 80 training networks contain 3,121 weak components in total; student test contains 227. Reference sequences have 13 training endpoint breaks and one student-test break.'),
('DATA8','What are isolated nodes and weak components?','An isolated node has no incident road arcs; the audited 90 networks have none. A weak component is a connected region after ignoring road direction. Being in one weak component still does not guarantee a legal directed route.'),
('DATA9','Is the reference route perfect ground truth?','We cannot assume that. GPS and maps may disagree and references may contain breaks. We disclose those limitations rather than claiming independently verified physical truth.'),
('DATA10','Why keep file hashes and an inventory?','They identify the exact input files and record trip sizes, time gaps, geographic extent and network quality. One minor retained-record gap is that the inventory does not store each file\'s byte size.'),
]:qa(*item)

page('2. Preprocessing and features','Deliverables.pdf 2.2 questions 1-2; features.py; reference.py; prepare.py')
for item in [
('D-EDA1','How did you preprocess the data?','We validate shapes, finite coordinates, integer IDs, node bounds and increasing time. We project into local metres, generate candidates and derive separate training proxy labels. Invalid files raise errors; we do not silently impute GPS positions.'),
('D-EDA2','What patterns and features did you observe and choose?','Trips differ greatly in duration and map size; GPS is mostly 1 Hz but has long gaps. We use distance, direction, motion, edge geometry and junction degree because these describe whether a road fits an observation without knowing the reference route.'),
('FEAT1','Why not measure Euclidean distance in degrees?','A degree of longitude changes physical length with latitude. We use a local equirectangular projection: x = R cos(origin latitude) times longitude difference in radians; y = R times latitude difference. R is 6,371,008.8 m.'),
('FEAT2','Is this projection exact everywhere?','No. It approximates a local flat map. The largest audited network dimension is about 77.6 km. We disclose projection error as a limitation rather than using this formula for global distances.'),
('FEAT3','How are eight candidate roads found?','Every directed arc is sampled at most 30 m apart. A 115 m sample-radius search covers arcs intersecting the 100 m GPS radius. Exact point-to-segment projection ranks them, and we keep up to eight.'),
('FEAT4','Why does the index use 115 m rather than 100 m?','A segment point can lie halfway between index samples. That is at most 15 m with 30 m spacing, so the larger lookup radius avoids missing an arc before exact segment-distance filtering.'),
('FEAT5','What if there are no roads inside the radius?','Use a flagged nearest-sample fallback, then exact projection. This fallback is approximate and can be far away. We never insert a reference road to make candidate recall look better.'),
('FEAT6','What are all 13 features?','Distance; log distance; distance above the nearest candidate; signed heading cosine; absolute heading cosine; edge length; log length; projection fraction; speed; log previous time gap; source indegree; target outdegree; stationary indicator.'),
('FEAT7','How are heading and speed estimated?','From a local +/-3-observation window, avoiding gaps over 60 s. This uses future GPS points, so our method is offline. Speed is metres per second; heading cosine compares movement direction with directed-road orientation.'),
('FEAT8','What enters the model and what does not?','A feature vector for each observation/candidate pair enters. Arc IDs are metadata. Absolute reference labels, reference membership and route-derived diagnostics never enter prediction features.'),
]:qa(*item)

page('3. Labels, fitting and leakage','reference.py; training_manifest.json; folds.json')
for item in [
('LAB1','How do we get one proxy label per GPS observation?','We find a minimum-distance alignment that moves forward through the ordered reference arc sequence. A GPS observation can stay at the same route position or skip forward; it cannot move backwards in route order.'),
('LAB2','What is the label-alignment recurrence?','D[t,j] = squared distance from GPS t to route arc j + min over i <= j of D[t-1,i]. Prefix minima make this O(T times route length). Repeated visits keep separate route positions.'),
('LAB3','Is alignment another prediction input?','No. It is used only for supervised training labels and separate server-side evaluation after prediction. The prediction object has track, nodes and arcs, with no reference-route field.'),
('LAB4','Why exclude labels farther than 75 m during training?','They are weak supervision: the reference projection is far from the GPS. The fixed filter excludes 21,390 of 208,417 training observations before stride selection. All student-test observations remain scored, including 21 beyond 75 m.'),
('LAB5','How much data is actually fitted?','We select original observation indices divisible by five and apply the fixed label-quality mask. This yields 255,024 candidate rows and 36,942 positives across all 80 training trips. We do not claim every training GPS point was fitted.'),
('LAB6','What is a positive or negative candidate?','Positive means its directed arc ID equals that observation\'s inferred reference arc. Other candidates are negative. If the reference arc is absent, all candidates are negative; the miss is retained, not repaired.'),
('VAL1','Why use trip-grouped CV?','Nearby observations from one trip are strongly related. Splitting rows randomly would leak trip-specific patterns. Three StratifiedGroupKFold folds keep whole trips together while trying to balance the binary labels.'),
('VAL2','Where is scaling fitted?','Inside the logistic and Gaussian NB pipelines, separately on each fold\'s training rows. Validation and student-test rows only receive the fitted transform. Tree models do not need that standardisation.'),
('VAL3','Why disable boosting\'s automatic early stopping?','Its default internal validation can split candidate rows from one trip across fitting and validation. We disable it so the explicit whole-trip folds define validation.'),
('VAL4','Can CV and student-test results prove unbiased final accuracy?','No. CV selects/tunes the classifier, and student test selects decoder settings. They are development estimates. Reserved evaluation is separate and the predictor was frozen before those files were opened.'),
]:qa(*item)

page('4. Metric questions you must distinguish','metrics.py; experiments.py; evaluation.py')
for item in [
('MET1','What is accuracy and observation Top-1 accuracy?','Candidate accuracy counts correct binary candidate decisions. Top-1 chooses one highest-scoring road per observation and checks its directed arc against the proxy reference label. These have different denominators.'),
('MET2','What are precision and recall?','Precision = TP/(TP+FP): among roads labelled a match, how many are positive? Recall = TP/(TP+FN): among positive candidates, how many were detected? They reveal different kinds of errors.'),
('MET3','What is Macro F1?','F1 = 2PR/(P+R). Candidate Macro F1 averages the F1 for binary classes 0 and 1 equally. Arc Macro F1 averages over trip-local road classes. Neither is the same as route-overlap F1.'),
('MET4','Why use candidate Macro F1 for selection?','Most candidate rows are negative, so plain candidate accuracy can look good while missing positives. Macro F1 balances the two classes. Its 0.5 threshold objective is still different from within-observation ranking, which is a limitation we measure.'),
('MET5','What does a confusion matrix show?','Rows are true candidate labels, columns are predicted labels. In order 0,1 it contains TN/FP on the first row and FN/TP on the second. It makes false matches and missed matches visible.'),
('MET6','What is log loss and why use it?','It punishes assigning a low probability to the true label; confident wrong answers hurt more. Lower is better. We report binary candidate log loss separately from observation-level proxy loss after normalising candidate scores.'),
('MET7','What is candidate miss rate?','The fraction of GPS observations whose proxy reference arc is absent from the candidate set. Student-test miss rate is 0.3601%. With that fixed candidate set, even a perfect ranker cannot exceed about 99.64% proxy Top-1.'),
('MET8','How is route-overlap F1 computed?','Compare sets of unique directed predicted/reference arcs, weighted by arc length in metres. Precision measures how much predicted road length belongs to the reference; recall measures how much reference road length was recovered.'),
('MET9','What do continuity and ordered LCS F1 measure?','Direct continuity checks whether consecutive distinct predicted arcs meet at endpoints. Ordered LCS F1 compares ordered collapsed arc sequences and penalises ordering/repetition differences. Overlap alone ignores repeated use of the same arc.'),
('MET10','Why can two tables have different averages or times?','Pooled Top-1 weights GPS observations, while mean-trip metrics give each trip equal weight. Model-scoring time excludes preprocessing/decoding; pipeline time includes them; server time also includes scoring/map work. Runs and hardware differ.'),
]:qa(*item)

page('5. Five-model comparison at a glance','model_comparison.json; selection.json; ui/details.html')
rows=[['Model','CV cand. F1','Dev Top-1','Dev cand. F1','Arc F1*','Cand. loss']]
for m in MODELS:
 s=m['student_test']['summary'];rows.append([SHORT[m['name']],f"{m['cv_candidate_macro_f1']:.4f}",pct(s['proxy_top1_accuracy']),f"{s['candidate_macro_f1']:.4f}",f"{s['mean_trip_proxy_arc_macro_f1']:.4f}",f"{s['candidate_log_loss']:.4f}"])
table(rows,[110,76,76,76,76,76])
p('*Arc F1 is mean-trip proxy arc Macro F1. These are untuned baseline comparisons, not the final HMM route metrics.',True)
qa('SEL1','Which model was selected, and why?','Histogram gradient boosting had the highest mean grouped-CV candidate Macro F1, 0.9278, versus random forest 0.9260. Student-test performance did not select the family. The small difference does not prove statistical superiority.')
qa('SEL2','Why not choose the highest training accuracy?','A flexible model can memorise training data. Grouped validation tests transfer to unseen trips; training performance alone cannot establish generalisation.')
qa('SEL3','Which graph is appropriate for every model?','We show measured CV/development comparisons and per-model confusion/generalisation plots. We do not invent neural-network-style epoch curves for models that did not produce those histories.')
qa('SEL4','Why are KNN, SVM, neural networks and clustering absent?','They were examples, not compulsory choices. Five distinct fitted models meet the count. We chose CPU-friendly tabular baselines; we do not claim to have trained untested model families.')
qa('SEL5','Would a tiny CV lead justify a strong claim?','No. Boosting\'s untuned fold SD is about 0.0104, larger than its mean gap to random forest. We use the declared selection rule, acknowledge close alternatives, and report the fold values.')

EXPLAIN={
'logistic':('A simple linear baseline checks whether distance and direction are already enough.','It combines weighted features into a score, then applies a sigmoid to get a binary match probability.','It is compact and fast, but a linear boundary cannot express all junction and motion interactions.','L2-regularised binary log loss, solved by L-BFGS; scaling occurs inside each fold.','C controls inverse regularisation: smaller C means stronger regularisation. max_iter limits optimisation steps.'),
'gaussian_nb':('A fast probabilistic baseline tests a different assumption about feature distributions.','For each class it fits Gaussian feature likelihoods and combines them with the class prior using Bayes\' rule.','It fits cheaply, but correlated distance/log-distance and heading features violate conditional independence; probabilities can become overconfident.','Gaussian maximum likelihood: choose class-wise means and variances that best explain the training features, with empirical class priors. Variance smoothing stabilises tiny variances; there is no iterative loss curve.','Naive means features are treated as independent given the class, not that features are independent in the whole dataset.'),
'decision_tree':('A bounded tree can learn nonlinear thresholds and feature interactions.','It splits the feature space into regions and uses the class proportions in each final leaf for probabilities.','Threshold rules are understandable, but one tree can be unstable and its probabilities are coarse.','Greedy reduction in Gini impurity, limited by depth and minimum leaf size.','max_depth limits the number of splits along a path; min_samples_leaf avoids leaves based on too few candidate rows.'),
'random_forest':('Averaging trees tests whether bagging improves stability over one tree.','Trees learn from bootstrap samples and feature subsets; their predicted match probabilities are averaged.','Averaging reduces variance, but fitting and model size cost more. Correlated trees and weak labels still limit generalisation.','Bootstrap aggregation of trees with Gini splits; probabilities are averaged across leaves.','Bagging trains trees in parallel on resampled data; boosting builds successive trees to correct the current model.'),
'hist_gradient_boosting':('Boosting can learn nonlinear feature interactions and correct systematic candidate errors efficiently.','Features are binned into histograms. Successive shallow trees improve the current prediction using loss gradients.','It gave the highest baseline CV score at modest CPU cost; tuning and weak proxy labels can still cause overfitting.','Binary log loss with L2 leaf regularisation. Automatic row-random early stopping is disabled.','Lower learning rate takes smaller updates; leaf count controls tree complexity; L2 and minimum leaf size regularise the fit.'),
}
for model_number,m in enumerate(MODELS,1):
 name=m['name'];why,how,limits,objective,follow=EXPLAIN[name];s=m['student_test']['summary'];gap=m['training']['candidate_macro_f1']-m['cv_candidate_macro_f1']
 page(f'6.{model_number} '+NAMES[name],'Deliverables.pdf 3.2: A1/A2, B5/B6, C7/C8/C9; models.py; model_comparison.json')
 qa(name+'-A1','Why did you consider this algorithm suitable?',why+' '+how)
 qa(name+'-A2','What are its major strengths and limitations for this task?',limits)
 qa(name+'-B5','What were the initial hyperparameters?',json.dumps(m['parameters'],sort_keys=True)+'. Seed 357 where applicable. These are the untuned baseline settings.')
 qa(name+'-B6','What loss or optimisation objective was used?',objective)
 qa(name+'-C7','How did it perform on the student-testing dataset?',f"Proxy Top-1 {pct(s['proxy_top1_accuracy'])}; candidate precision {s['candidate_precision']:.4f}, recall {s['candidate_recall']:.4f}, candidate Macro F1 {s['candidate_macro_f1']:.4f}. Mean route-overlap F1 {s['mean_trip_route_length_f1']:.4f}; direct continuity {s['mean_trip_direct_arc_continuity']:.4f}.")
 qa(name+'-C8','What are its Top-1, Macro F1 and log-loss values?',f"Top-1 {pct(s['proxy_top1_accuracy'])}; mean-trip arc Macro F1 {s['mean_trip_proxy_arc_macro_f1']:.4f}; candidate Macro F1 {s['candidate_macro_f1']:.4f}; candidate log loss {s['candidate_log_loss']:.4f}; observation proxy loss {s['observation_proxy_log_loss']:.4f}. Keep the metric names separate.")
 qa(name+'-C9','Did you observe overfitting or underfitting? What evidence supports it?',f"Training candidate Macro F1 is {m['training']['candidate_macro_f1']:.4f}, versus CV {m['cv_candidate_macro_f1']:.4f}, a gap of {gap:.4f}. "+('This is a modest training advantage consistent with overfitting, not proof of its cause.' if gap>.02 else 'There is no large fit/CV gap on this metric; that does not prove perfect generalisation or rule out underfitting.'))
 qa(name+'-F1','What is one likely follow-up?',follow)
 p(f"Measured full-fit time: {m['full_fit_seconds']:.3f} s. Model-only scoring across the ten student-test trips: {s['model_inference_seconds']:.3f} s. These timings belong to the original Linux run; they are not universal hardware benchmarks.",True)

page('7. Bayesian hyperparameter optimisation','Deliverables.pdf Section 4 questions 1-6; tuning.json; bayesian_trials.json')
qa('D-BO1','Which hyperparameters were included in the search?','For the selected boosting family: learning_rate, max_leaf_nodes, l2_regularization and min_samples_leaf. max_iter remains fixed at 120.')
qa('D-BO2','What ranges or distributions were considered?','Learning rate: log scale 0.025-0.2. L2: log scale 0.01-10. Leaf count: integers 7-31. Minimum leaf size: integers 15-100. This spans update size, complexity and regularisation within a bounded CPU search; it is not a claim of exhaustive optimisation.')
qa('D-BO3','What objective function was optimised?','Mean three-fold trip-grouped candidate Macro F1 at the binary 0.5 threshold, using only the 80 training trips. Each trial uses the same fold assignments.')
qa('D-BO4','How many optimisation trials were performed?','Twelve sequential trials with seed 357: four random startup trials, then eight adaptive TPE trials. The recorded search took about 69.6 seconds on that run.')
qa('D-BO5','What were the best values?',f"learning_rate {TUNING['best_parameters']['learning_rate']:.6f}; max_leaf_nodes 18; l2_regularization {TUNING['best_parameters']['l2_regularization']:.6f}; min_samples_leaf 60. Best CV objective {TUNING['cv_candidate_macro_f1']:.6f}.")
qa('D-BO6','Did the tuned model outperform the untuned model?','On the declared CV objective, yes: 0.9278 to 0.9290. On student-test proxy Top-1, no: 87.63% to 87.46%. Candidate loss also slightly worsened. We report both, rather than claiming tuning improves everything.')
u=TUNING['untuned_student_test']['summary'];t=TUNING['tuned_student_test']['summary']
table([['Same ten student-test trips','Untuned','Tuned'],['Candidate Macro F1',f"{u['candidate_macro_f1']:.4f}",f"{t['candidate_macro_f1']:.4f}"],['Mean-trip arc Macro F1',f"{u['mean_trip_proxy_arc_macro_f1']:.4f}",f"{t['mean_trip_proxy_arc_macro_f1']:.4f}"],['Candidate log loss',f"{u['candidate_log_loss']:.4f}",f"{t['candidate_log_loss']:.4f}"],['Model-only scoring seconds',f"{u['model_inference_seconds']:.3f}",f"{t['model_inference_seconds']:.3f}"]],[300,95,95])
qa('BO7','What makes TPE adaptive rather than pure random search?','After startup, TPE models settings associated with good and other objective values and proposes promising settings. Trials use past outcomes; they do not inspect reserved-trip labels.')
qa('BO8','Are hyperparameters the learned weights?','No. Hyperparameters control fitting, such as leaf size or learning rate. Model parameters are learned from training data. TPE searches hyperparameters; each trial fits a model.')

page('8. Required HMM questions','Deliverables.pdf Section 5 questions 1-6; decoder.py; geometry.py')
qa('D-HMM1','What are the hidden states?','At each GPS observation, a state is a candidate directed arc with a projected position on that arc. Keeping the arc ID and projection fraction preserves the state-to-road correspondence.')
qa('D-HMM2','How are emissions calculated?','The classifier gives P(match) for each valid candidate. We divide each positive score by the sum across candidates. Invalid padding gets zero; tiny valid scores are floored at 1e-12 for numerical stability. These are discriminative emissions, not a separately fitted GPS-noise density.')
qa('D-HMM3','How are transitions calculated?','Compute directed road distance between projected positions, including remaining start-arc length, shortest-path middle distance and end-arc prefix. The final log compatibility is -abs(road distance - GPS displacement)/25 m. Impossible transitions get negative infinity.')
qa('D-HMM4','How are missing candidates or large GPS gaps handled?','A radius miss gets an explicit nearest-sample fallback. A GPS gap greater than 60 s starts a new sequence, so the decoder does not imply a known route through the outage.')
qa('D-HMM5','What if consecutive observations suggest disconnected roads?','A transition is impossible when no directed route fits 55 m/s times elapsed seconds plus 100 m slack. If no complete path survives to the next step, Viterbi starts a new piece and returns a reset reason.')
qa('D-HMM6','Did HMM decoding improve the final route? Support your answer.','On student test, bridged overlap F1 improves 0.9339 to 0.9570 and direct continuity 0.7432 to 0.9961. Proxy Top-1 falls 87.46% to 85.42%. On reserved data, overlap improves 0.9160 to 0.9250 while proxy Top-1 falls 88.44% to 77.78%. Improvement depends on the metric.')
qa('HMM7','Why not use straight-line distance between roads?','Road direction, detours and connectivity matter. Two roads may be physically close but unreachable without a long detour. The graph distance captures this; straight-line proximity alone does not.')
qa('HMM8','Are our transition values normalised probabilities?','The final configuration uses globally scaled exponential compatibility potentials, not candidate-row-normalised probabilities. It is an HMM-style max-product decoder with classifier emissions. This wording is more precise than claiming a fully learned generative HMM.')
qa('HMM9','Why can the same arc require a loop?','A backwards projection over 3 m cannot travel backwards along that directed arc. A real directed loop may be needed. Reconstruction retains that loop; up to 3 m is treated as stationary GPS jitter.')

page('9. Viterbi made easy','decoder.py; brute-force and same-arc-loop tests')
qa('VIT1','Why not pick the best road independently at every point?','The top local road can change to an incompatible road at the next point. Viterbi compares whole candidate sequences, trading local match scores against movement/connectivity compatibility.')
qa('VIT2','What is the Viterbi recurrence?','delta[t,j] = log emission[t,j] + max over previous i of (delta[t-1,i] + log transition[i,j]). Store the winning predecessor for each j, then backtrack the best final state.')
qa('VIT3','Why work in log space?','Multiplying many small probabilities can underflow to zero. Logs turn products into sums, keeping scores numerically manageable. Impossible paths are represented by negative infinity.')
qa('VIT4','Does subtracting the maximum path score change the answer?','No. At one time step, subtracting the same constant from all surviving scores preserves their order and the best path. Our decoder does this to keep values bounded.')
p('<b>Tiny teaching example - not an experiment or a real trip:</b> first emissions A=0.8, B=0.2; second emissions A=0.3, B=0.7. Transitions A-to-A=0.9, A-to-B=0.1, B-to-A=0.2, B-to-B=0.8.')
table([['Path','Product score'],['A -> A','0.8 x 0.9 x 0.3 = 0.216'],['A -> B','0.8 x 0.1 x 0.7 = 0.056'],['B -> A','0.2 x 0.2 x 0.3 = 0.012'],['B -> B','0.2 x 0.8 x 0.7 = 0.112']],[160,330])
p('Independent emission picks are A then B. The best full path is A then A, because 0.216 is largest. This is why a decoder can override the classifier.')
qa('VIT5','What are its time and memory costs?','With T observations and K candidates, the recurrence costs O(T K squared); here K is at most eight. Backpointers cost O(T K). Actual runtime also includes shortest-path routing, stored transition matrices and bridge reconstruction.')
qa('VIT6','What does bridge reconstruction add?','Viterbi selects one arc per GPS point. GPS can skip short roads between points, so reconstruction inserts actual shortest-path intermediate arcs. It does not draw a fabricated straight connector across a reset.')
qa('VIT7','How was correctness tested?','A tiny Viterbi example is checked against exhaustive enumeration of all paths. Other tests cover resets, directed partial-arc distance, candidate projection and preservation of a required same-arc loop. These tests do not prove high accuracy on every real trip.')

page('10. Actual results and difficult questions','decoder_comparison.json; evaluator_results.json')
table([['Metric','Dev classifier','Dev HMM/bridge','Reserved classifier','Reserved HMM/bridge'],['Proxy pointwise accuracy','87.46%','85.42%','88.44%','77.78%'],['Mean overlap F1','0.9339','0.9570','0.9160','0.9250'],['Mean direct continuity','0.7432','0.9961','0.7604','0.9976'],['Mean ordered LCS F1','0.7332','0.8012','0.7630','0.7547']],[118,93,93,93,93])
qa('RES1','Why does higher continuity not prove better accuracy?','A route can be perfectly connected while following the wrong road or taking extra loops. Bridge insertion itself raises continuity. HMM-state continuity before bridges is only 0.8822 on development and 0.8440 on reserved trips.')
qa('RES2','Why might HMM make pointwise results worse?','Its movement assumptions, truncated candidate set, proxy-label ambiguity and map defects can disagree with local classifier rankings. Those are possible mechanisms, not experimentally proven explanations for every failure.')
qa('RES3','Does one easy demo show the overall performance?','No. Student trip 00000052 reaches about 99.62% HMM proxy Top-1, while 00000077 is about 64.41%. Reserved trip 00000015 is 48.91% with 104 resets. All ten trips must work, including poor-scoring ones.')
qa('RES4','Does fixing a code bug justify changing reported numbers?','Yes, if it is documented and results are recomputed honestly. Before reserved access, same-arc-loop reconstruction was corrected. Development bridged overlap became 0.9570, replacing the incomplete-bridge figure 0.9648; pointwise predictions did not change.')
qa('RES5','Are missing candidates the only source of error?','No. Even with the right road in the set, the model can rank it wrongly or the decoder can select another path. The reference label itself may be uncertain. Candidate recall is only one part of the pipeline.')
qa('RES6','Is a predicted route length a GPS distance or exact physical travel distance?','No. The trace sums the full lengths of reconstructed arcs, including repeated traversals. Candidate states lie partway along edges, and map defects/gaps add uncertainty. Treat it as reconstructed network length, not a precise odometer reading.')
qa('RES7','What do we say if asked for a novelty?','No new scientific method is claimed for the required baseline. Bhuvan\'s extension is pending. Standard HMM decoding, TPE tuning and a useful trace UI must not be renamed as a novel research algorithm without evidence.')

page('11. Chiranth\'s pushed UI and trace code','ui/index.html; ui/details.html; api.py; trace.py; commit 3d05683')
qa('CHI1','What did Chiranth add?','A map-first evaluator screen, a separate Model trace tab, per-observation candidate/transition explanations, a four-step model-selection display and a later reserved-trip comparison across all model families.')
qa('CHI2','How does the new map work?','Leaflet draws the actual predicted route geometry. Toggles show raw GPS, classifier-only segments and reset markers. GPS is drawn above the road route so both remain visible. Older APIs fall back to the server-generated Folium map.')
qa('CHI3','What are the four model-selection steps on the page?','Train five models, choose the highest grouped-CV candidate Macro F1, tune the selected family with TPE, then compare tuned/untuned student-test performance. This displayed history is separate from the live trip inference pipeline.')
qa('CHI4','What are the six live pipeline stages?','Read trip, project to metres, candidate search, classifier, Viterbi decoding, and route plus scoring. Stage cards show server-derived observation counts, distances, candidate pairs, timings, overrides and resets.')
qa('CHI5','What does GET /trace/{request_id} return?','A read-only serialisation of a finished prediction: GPS/route geometry, stage statistics, per-point candidates, distances, probabilities, emissions and classifier/Viterbi picks. It reads no reference route and cannot change predictions.')
qa('CHI6','What does GET /explain/{request_id}/{observation} show?','Features and scores for that point, plus road distances and transition terms from the selected previous Viterbi state. At a reset it gives the reason instead. Display values are rounded.')
qa('CHI7','Is its step score the full Viterbi score?','No. It shows local log emission plus log transition from one selected predecessor. Actual Viterbi compares all previous candidates and carries their accumulated path scores. The display is an explanation, not a second decoding algorithm.')
qa('CHI8','Is mean top P(match) the trip accuracy?','No. It averages the highest candidate match probability per point and is a confidence-like summary. Accuracy requires separate reference scoring. A high model score can still be confidently wrong.')
qa('CHI9','How do the main tab and trace tab stay in sync?','They share the latest request/session through browser BroadcastChannel with stored state. Both tabs should use the same UI origin. The trace tab fetches explanation data from the API only as needed.')
qa('CHI10','Why can a trace or explanation disappear?','The server keeps at most 32 traces/maps and eight full predictions. Older explanations expire to bound memory. Reconstruct the trip again; a missing trace does not mean the model was retrained.')
qa('CHI11','What is currently preventing the new trace on Linux?','At audit time, the running API schema lists only the old endpoints. Pulling changes does not reload an existing Python process. Restart the API to load trace.py and the new endpoints; the source itself passes 13 tests.')

page('12. Chiranth\'s later reserved comparison','evaluator_model_comparison.json; evaluate_reserved_models.py; ui/details.html')
p('<b>Reporting only:</b> these 19,674 reserved observations did not select the served model. The five original settings were fitted on training data for comparison; the frozen tuned artifact was loaded unchanged. The tuned row agrees with the earlier frozen evaluation.')
rows=[['Family / variant','Classifier Top-1','Arc Macro F1*','Class route F1','Class continuity','HMM Top-1']]
for m in LATER['models']:
 s=m['summary'];label=SHORT[m['name']]+(' tuned' if m['tuned'] else ' base' if m['name']=='hist_gradient_boosting' else '')
 rows.append([label,pct(s['classifier_proxy_top1']),f"{s['classifier_proxy_macro_f1_mean']:.4f}",f"{s['classifier_route_f1_mean']:.4f}",f"{s['classifier_continuity_mean']:.4f}",pct(s['hmm_proxy_top1'])])
table(rows,[105,77,77,77,77,77]);p('*Mean-trip proxy arc Macro F1. Top-1 pools observations. These are final-partition descriptive results, not CV scores.',True)
rows=[['Family / variant','Bridge precision','Bridge recall','Bridge F1','Bridge continuity']]
for m in LATER['models']:
 s=m['summary'];label=SHORT[m['name']]+(' tuned' if m['tuned'] else ' base' if m['name']=='hist_gradient_boosting' else '')
 rows.append([label,*[f"{s[k]:.4f}" for k in ['bridged_route_precision_mean','bridged_route_recall_mean','bridged_route_f1_mean','bridged_continuity_mean']]])
table(rows,[130,90,90,90,90])
rows=[['Family / variant','Fit s','Model-score s','Full prediction s']]
for m in LATER['models']:
 s=m['summary'];label=SHORT[m['name']]+(' tuned' if m['tuned'] else ' base' if m['name']=='hist_gradient_boosting' else '')
 rows.append([label,'Loaded artifact' if s['fit_seconds'] is None else f"{s['fit_seconds']:.3f}",f"{s['model_scoring_seconds']:.3f}",f"{s['total_prediction_seconds']:.3f}"])
table(rows,[160,110,110,110]);p('Times sum the ten reserved trips in the later run. They are not end-to-end browser/server times and should not be compared as controlled speedups against the original Linux training run.',True)
qa('CHI12','Why not switch to the model with the best reserved table score?','Because reserved results are final evaluation, not model selection. Switching based on them leaks final-test information. We retain the model chosen earlier by grouped training-trip CV.')
qa('CHI13','Are the static comparison tables hardcoded predictions?','No. They display recorded experiment summaries, like report figures. A live trip still goes through POST /match and the actual model/decoder. Static summaries must remain consistent with their source JSON artifacts.')

page('13. API, deployment and Folium questions','Main Guidelines.pdf Section 3; Models and Training Guidelines.pdf final workflow; Deliverables.pdf Sections 6-7')
for item in [
('API1','What runs on the two laptops?','Linux Laptop 1 hosts model, road inputs, decoder and server-side scoring. Chiranth\'s Mac runs the static UI in its browser. It sends the selected trip ID to Linux; it does not need the raw dataset or ML packages for UI-only operation.'),
('API2','What happens after the evaluator selects a trip?','The UI sends POST /match with trip_id. The server loads track/network, generates features, predicts, decodes and reconstructs. Only after prediction does it read .route to score. Results, a request ID and map/trace URLs return.'),
('API3','What are health, trips and request IDs for?','GET /health checks readiness/mode; GET /trips lists ten active IDs. The request ID links the displayed result to the receiving laptop\'s log, proving the selected request went through the API.'),
('API4','Why use the Linux LAN address rather than localhost on the Mac?','localhost always means the machine where the browser runs. On the Mac that points to the Mac. The Model server field must point to Linux, currently the LAN address ending :8000, while the local UI can run on :8010.'),
('API5','Why is CORS needed?','The Mac UI and API have different origins, including different ports/hosts. Browser CORS rules require the API to permit that cross-origin request. This LAN demo permits unauthenticated GET/POST requests without credentials.'),
('D-MAP1','How do map colours and markers distinguish layers?','Development Folium maps use blue GPS points, grey candidate roads, gold reference, green prediction, purple classifier-only and red resets. Final API maps omit the reference. The new UI also distinguishes/toggles GPS, prediction, classifier and resets.'),
('D-MAP2','How are labels, zoom and extent handled?','Folium has an explicit legend, scale, layer names and controls. Map bounds fit the GPS extent. Coordinates are swapped to latitude/longitude for display.'),
('D-MAP3','How can observed GPS and predicted roads be compared?','Overlay their distinct layers and zoom into deviations. Chiranth\'s UI draws GPS above the predicted road and offers toggles. The development report map can add reference geometry; evaluator UI must not.'),
('D-MAP4','Is Folium still mandatory if the new UI uses Leaflet?','Yes. We retain Folium code, committed interactive HTML maps and an actual Folium screenshot in the technical report. Leaflet is an additional evaluator interface, not a replacement for that required evidence.'),
('D-MAP5','What screenshots and final checks are required?','A current working UI screenshot and a genuine successful API/terminal screenshot on the receiving laptop. Verify the physical two-laptop route and matching request ID. Those final evidence steps are still open in the audit.'),
]:qa(*item)

page('14. Integrity, limits and likely examiner traps','Assessment and Plagiarism Policy.pdf; all five organiser PDFs; current audit')
for item in [
('INT1','What are the two official submission deliverables?','A public GitHub repository containing the full implementation/artifacts/results, and one consolidated technical-report PDF. This study PDF is rehearsal material; do not treat it as a second model report for submission.'),
('INT2','How do we prove evaluator data did not select the model?','Commit a61ce50 froze the model, settings and prediction/scoring source hashes before archive access. The reserved runner checks those hashes. Later UI/trace work is read-only; current core predictor hashes still match.'),
('INT3','Can evaluator labels appear in the UI or predictor?','No. Prediction inputs are track, nodes and arcs. Reference .route is read only for training labels or after predictions for scoring. API maps/traces return prediction data and aggregate scores, not reference arc labels/geometry.'),
('INT4','Does using AI automatically make the work invalid?','The policy requires actual experimentation, truthful attribution and team understanding. Codex assistance is disclosed. We must not present generated explanations as evidence of experiments that never happened or claim we understand code we cannot explain.'),
('INT5','Can we share model weights or implementation with another team?','No. Cross-team sharing of code, trained weights, configurations, derived datasets or results is prohibited. Collaborate only within the authorised team and follow the required public-repository submission process.'),
('INT6','Can an audit certify no plagiarism or that everyone understands?','No. We can inspect evidence, source and reproducibility. Originality and each member\'s technical understanding still need human review and a genuine viva/demonstration.'),
('INT7','Can we claim GPU/Kaggle training, online inference or a novel HMM?','No. The measured baselines ran on CPU, heading uses future GPS, and the decoder uses established ideas. Claim only work that was actually done and validated.'),
('INT8','What tests passed and what do they not establish?','Thirteen current tests pass, including geometry, grouped folds, Viterbi, reference separation, API behaviour and trace/prediction consistency. They do not replace physical two-laptop testing, real accuracy evaluation or team understanding.'),
('INT9','What are the remaining deliverable gaps?','Restart the API for the new trace endpoints; verify physical Mac-to-Linux use; capture the required receiving-terminal screenshot; refresh report screenshots for the final UI; combine family/settings into the requested overall comparison table. A file-size inventory is a minor exploration-record gap.'),
('INT10','What is the safest answer when we do not know a cause?','Say what the measurement shows, then separate a possible explanation from a verified result. For example: HMM accuracy falls on trip 15; reference ambiguity or transition assumptions could contribute, but this run did not isolate the cause.'),
]:qa(*item)

page('15. Official-question coverage and last rehearsal')
table([['Official requirement / questions','Where answered'],['Deliverables.pdf 2.2: preprocessing, patterns and chosen features','D-EDA1, D-EDA2; DATA1-DATA10; FEAT1-FEAT8'],['Deliverables.pdf 3.2, for EVERY model: A1/A2/B5/B6/C7/C8/C9','Each of the five model pages has all seven explicit IDs, plus a follow-up.'],['Deliverables.pdf 4: six Bayesian questions and tuned/untuned metrics','D-BO1 through D-BO6; BO7-BO8; comparison table'],['Deliverables.pdf 5: six HMM questions and measured comparison','D-HMM1 through D-HMM6; HMM7-HMM9; VIT1-VIT7; results'],['Deliverables.pdf 6: colours, labels, extent, comparison, layer controls','D-MAP1 through D-MAP5; API/Folium page'],['Main / Models guidelines: two-laptop flow, metrics, independence','API1-API5; MET1-MET10; SEL1-SEL5; INT1-INT10'],['Dataset / Assessment PDFs: quality, shapes, geography, leakage, viva','DATA / FEAT / LAB / VAL / INT pages'],['Chiranth\'s pushed code and later metrics','CHI1-CHI13 and the full six-variant reserved tables']],[245,245])
p('<b>Rehearsal in pairs:</b> one person asks a random question ID; the other gives a 20-second answer and points to its code/artifact. Swap roles. Everyone should explain one observation from GPS to candidate features, model score, Viterbi choice and server result.')
p('<b>Do not memorise only a good number.</b> Explain why 0.9290 is a CV candidate F1, why 77.78% is reserved HMM proxy accuracy, and why 0.9250 is a mean route-overlap F1. They answer different questions.')
p('<b>Project sources:</b> the five supplied organiser PDFs; AGENTS.md; docs/DELIVERABLES_AUDIT.md; source modules; the recorded JSON artifacts. Repository snapshot reviewed: main at 3d05683. The required submission PDF is reports/PathSnap_Final_Report.pdf.',True)
p('Primary technical references checked: scikit-learn 1.7.2 cross-validation and model-evaluation documentation; Optuna TPE reference; Newson and Krumm (2009), Hidden Markov Map Matching Through Noise and Sparseness. These explain established methods; all project numbers come from our own artifact files.',True)
for label,url in [('Grouped CV','https://scikit-learn.org/1.7/modules/cross_validation.html'),('Metric definitions','https://scikit-learn.org/1.7/modules/model_evaluation.html'),('TPE','https://optuna.readthedocs.io/en/v4.6.0/reference/samplers/generated/optuna.samplers.TPESampler.html'),('Map matching paper','https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/')]:
 p(escape(label+': '+url),True)

required=['D-EDA1','D-EDA2']+[f'D-BO{i}' for i in range(1,7)]+[f'D-HMM{i}' for i in range(1,7)]+[f'D-MAP{i}' for i in range(1,6)]
required += [m['name']+'-'+q for m in MODELS for q in ['A1','A2','B5','B6','C7','C8','C9']]
assert set(required)<=set(ids)
output=ROOT/'reports/supporting/PathSnap_Viva_Study_Guide.pdf'
def footer(canvas,doc):
 canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#6a7b70'))
 canvas.drawString(43,24,'PathSnap | PT-088 | Viva preparation, not a submission report')
 canvas.drawRightString(A4[0]-43,24,str(doc.page));canvas.restoreState()
SimpleDocTemplate(str(output),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=39,bottomMargin=43,title='PathSnap Viva Study Guide - PT-088',author='PT-088 - AI-assisted study material').build(story,onFirstPage=footer,onLaterPages=footer)
manifest={'question_count':len(ids),'question_ids':ids,'official_explicit_coverage_count':len(required),'official_explicit_ids':required,'sections':sections,'audited_code_commit':'3d05683','note':'Audit and study material only; no frozen predictor or report edits.'}
(ROOT/'artifacts/viva_study_coverage.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Created {output}; {len(ids)} questions, {len(required)} mandatory question/checklist checks.')
