# Causal Design Map For Marketing Analytics

**Estimated time: 25-35 minutes**

CausalPy's example gallery is organized by design family: panel data methods
such as synthetic control, geolift, difference-in-differences, and interrupted
time series, followed by cross-sectional methods such as regression
discontinuity, instrumental variables, and inverse propensity weighting.

For this course, treat those examples as a menu of commercial measurement
designs. Start with the business constraint, identify how treatment was
assigned, and then choose the method whose assumptions are plausible.

## Panel And Time-Series Designs

| CausalPy reference | Marketing or commercial question | Use when | Main assumption |
|---|---|---|---|
| Interrupted time series | Did a national TV, PR, or pricing intervention move revenue above baseline? | There is one treated time series and no credible untreated control group. | The pre-period model would have continued to predict the post-period counterfactual. |
| Fixed-period interrupted time series | What was the lift during a campaign flight, and did it decay after the campaign ended? | Treatment has clear start and end dates. | Post-campaign behavior can be separated from in-campaign lift. |
| Synthetic control | What would a treated market have done without the campaign? | There are untreated donor geos with enough pre-period history. | A weighted donor combination can reproduce the treated unit before treatment. |
| Bayesian geolift | What is the campaign lift in a treated geo, including uncertainty? | You want a full uncertainty distribution for geolift. | Donor geos remain valid controls during treatment. |
| Multi-cell geolift | What was the average lift across multiple treated geos? Which treated geo moved most? | Multiple regions received the same or related treatment. | Pooled and unpooled estimands are not interchangeable. |
| Difference-in-differences | Did a regional rollout or store policy change affect sales? | There are treated and untreated units before and after rollout. | Treated and control units would have followed parallel trends without treatment. |
| Comparative interrupted time series | Did a treated time series change relative to a comparison series? | You have a comparison unit but treatment assignment was not randomized. | The comparison series captures untreated shocks and trends. |
| Piecewise interrupted time series | Did a growth rate or slope change after a launch? | The effect is expected to change trend, not just level. | The breakpoint is known or justified before analysis. |

## Cross-Sectional Designs

| CausalPy reference | Marketing or commercial question | Use when | Main assumption |
|---|---|---|---|
| Regression discontinuity | Did eligibility for free shipping, a discount, or a loyalty tier change conversion at the threshold? | Treatment is assigned by a numeric cutoff. | Units just below and just above the cutoff are comparable except for treatment. |
| Donut regression discontinuity | Does the threshold effect remain after excluding observations that may be manipulated near the cutoff? | Customers or sales teams can bunch just around the rule. | Excluding the donut removes the most suspicious observations. |
| Regression kink | Did marginal behavior change when the incentive rate changed? | A policy changes slope at a threshold rather than switching treatment on/off. | The running variable is smooth around the kink. |
| Instrumental variables | What is the causal effect of impressions, price, or sales contact when exposure is endogenous? | There is a shock or assignment rule that moves treatment but does not directly move the outcome. | The instrument is relevant and affects the outcome only through the treatment. |
| Weak-instrument diagnostics | Is the proposed instrument strong enough and defensible enough to use? | IV estimates are unstable or there are multiple candidate instruments. | Instrument relevance and exclusion can be argued separately. |
| Variable-selection priors for IV | Which instruments deserve more skepticism? | There are many candidate instruments with unequal credibility. | Prior structure reflects defensible domain knowledge. |
| Inverse propensity weighting | What is the average effect of an offer, contact, or campaign when targeting was observational? | Treated customers differ systematically from untreated customers. | All confounders needed for treatment selection and outcome are observed. |
| ANCOVA for nonequivalent groups | Did a treatment group improve after controlling for baseline performance? | There is a pre/post measure but the groups were not randomized. | Baseline outcome and covariates adequately adjust group imbalance. |

## Detailed Method Notes

The table above is the quick routing layer. The notes below explain what each
method is estimating, how the counterfactual is built, and what students should
check before turning the estimate into a marketing recommendation.

### Interrupted Time Series

Interrupted time series estimates the effect of an intervention by modeling a
single outcome series before the intervention, then projecting what would have
happened after the intervention if the pre-period relationship had continued.
The counterfactual usually comes from a regression, state-space model, Gaussian
process, or Bayesian time-series model using time trend, seasonality, known
control variables, and sometimes lagged outcomes.

In marketing, ITS is useful for national campaigns, PR events, brand launches,
website launches, pricing interventions, or marketplace policy changes where no
clean control market exists. The estimand is usually cumulative incremental
revenue, orders, leads, or conversion during the post-intervention window.

The main risk is that another event changed the outcome at the same time. A TV
campaign that starts during a competitor stockout may look stronger than it is.
Students should inspect pre-period fit, forecast error, residual seasonality,
known calendar shocks, and whether the intervention date was chosen before
looking at the data.

### Fixed-Period Interrupted Time Series

Fixed-period ITS is a version of interrupted time series where the campaign has
clear start and end dates. Instead of asking only whether the level changed
after launch, it separates in-campaign lift from post-campaign decay or
reversion. This matters for media flights because the effect may be temporary:
the business may see a lift during the campaign and a smaller residual effect
after the flight ends.

Use it for short campaigns, promotional bursts, creative launches, email
sequences, influencer campaigns, or seasonal pushes. The output should separate
total in-flight lift, post-flight lift, and any evidence that demand was pulled
forward rather than created.

Diagnostics should include the same checks as ITS, plus sensitivity to the
chosen campaign window. If the estimate changes dramatically when the window is
expanded by one or two weeks, the analyst should explain whether that is
expected lag, a measurement artifact, or noise.

### Synthetic Control

Synthetic control builds a counterfactual treated unit from a weighted
combination of untreated donor units. The weights are learned in the pre-period
so that the synthetic unit closely tracks the treated unit before treatment.
Those same weights are then applied in the post-period. The gap between the
treated unit and the synthetic unit is interpreted as treatment effect.

In commercial analytics, the treated unit may be a region, DMA, store group,
country, app market, or business line. Donor units are similar places or
segments that did not receive the campaign. This method is useful when a single
large market received a campaign and simple before/after comparison would be
confounded by seasonality or business momentum.

The most important diagnostic is pre-period fit. If the synthetic control
cannot reproduce the treated market before the campaign, the post-period gap is
not credible evidence. Students should inspect donor weights, pre-period MAPE
or RMSE, placebo tests on untreated donors, and whether any donor was partially
contaminated by the campaign.

### Bayesian GeoLift

Bayesian geolift applies the same basic counterfactual logic as synthetic
control or structural time series, but it returns a posterior distribution for
the lift. Instead of only reporting one gap estimate, the analyst reports a
range of plausible incremental outcomes and a probability that lift was
positive.

This is especially useful for stakeholder decisions because marketing tests are
rarely perfectly powered. A Bayesian geolift can say, for example, that the
campaign likely generated between 700 and 1,400 incremental orders, with 92
percent posterior probability of positive lift. That language maps naturally to
risk-aware rollout decisions and MMM calibration priors.

The key checks are donor validity, pre-period fit, posterior predictive checks,
and clarity about the prior. A narrow posterior from a bad control set is false
confidence. A wide posterior from a noisy test may still be useful if it
prevents overclaiming.

### Multi-Cell GeoLift

Multi-cell geolift estimates lift when several geos or cells receive treatment.
The analyst can estimate one pooled average effect, separate effects for each
treated cell, or a hierarchical model that shares information across cells
while allowing local differences.

Use this when a campaign is launched in multiple test markets, when several
regions receive different spend levels, or when a retailer runs a store-group
test across multiple territories. The business question should decide the
estimand: pooled lift answers "did the campaign work on average?", while
cell-level lift answers "where did it work best?"

Students should not mix these estimands casually. A strong lift in one region
can hide weak performance elsewhere. Diagnostics should include pre-period fit
by cell, heterogeneity across cells, pooled vs unpooled estimates, spillover
risk, and whether treated cells are representative of the rollout population.

### Difference-In-Differences

Difference-in-differences compares the before/after change for treated units to
the before/after change for untreated units. Its core idea is simple: if
treated stores increased by 8 percent after a loyalty offer and control stores
increased by 3 percent over the same period, the DiD estimate is a 5 percent
incremental effect.

Marketing examples include regional product launches, store-level sales
enablement programs, channel partner incentives, local promotions, CRM rollouts,
or phased feature launches. DiD works best when there are many treated and
control units observed over multiple periods.

The key assumption is parallel trends: without treatment, treated and control
units would have moved similarly. Students should plot pre-trends, run an
event-study check, cluster standard errors at the unit level, and look for
staggered rollout complications. If treated stores were selected because they
were already accelerating, the DiD estimate is likely biased upward.

### Comparative Interrupted Time Series

Comparative ITS is a bridge between interrupted time series and
difference-in-differences. It uses a comparison series to adjust the treated
series after an intervention, but it does not require a large panel of treated
and control units. The comparison series may be a similar region, a related
product category, a competitor benchmark, or a non-targeted business segment.

Use it when one business line or country receives a campaign and another
business line or country is not perfectly comparable but still captures shared
seasonality or macro shocks. The method asks whether the treated series changed
more than expected relative to the comparison.

Students should explain why the comparison series is useful but imperfect.
They should inspect pre-period co-movement, check for comparison-unit
contamination, and avoid claiming the same strength of evidence as a randomized
geo test.

### Piecewise Interrupted Time Series

Piecewise ITS, also called segmented regression, estimates changes in level and
slope around a known breakpoint. A standard ITS might ask "did revenue jump
after launch?" Piecewise ITS can ask "did the growth rate change after launch?"

This is useful for pricing changes, sales-process changes, SEO migrations,
retention-program launches, product-led-growth motions, and marketplace policy
changes where the effect accumulates over time. The short-term effect may be
small, but the slope change can be commercially meaningful.

The breakpoint should be justified before modeling. Students should check
whether the apparent slope change is sensitive to the chosen date, whether the
pre-period is long enough to estimate baseline trend, and whether other
business changes occurred around the same time.

### Regression Discontinuity

Regression discontinuity estimates a local causal effect at a threshold. Units
just below and just above the cutoff are assumed to be comparable except that
one side receives treatment. The model fits the outcome as a smooth function of
the running variable, allowing a jump at the cutoff.

Commercial examples include free-shipping thresholds, discount eligibility,
loyalty tier cutoffs, lead-score routing rules, account-size thresholds for
sales assignment, and credit limits. The estimand is local: the effect for
customers near the threshold, not all customers.

Students should plot the outcome around the cutoff, test multiple bandwidths,
allow different slopes on either side, and inspect whether observations bunch
just above or below the threshold. If customers or sales reps can manipulate the
running variable, the design becomes weaker.

### Donut Regression Discontinuity

Donut RD is a robustness check for regression discontinuity. It removes
observations very close to the cutoff, creating a "donut hole" around the
threshold. This is useful when the closest observations may be manipulated or
measured with error.

For example, customers may add a low-value item to cross a free-shipping
threshold, or account executives may round estimated account value so a lead
qualifies for a higher-touch sales motion. Removing the narrow band around the
threshold tests whether the estimated effect survives after excluding the most
suspicious observations.

The trade-off is precision. Removing observations reduces sample size and makes
the estimate less local. Students should report both the original RD and donut
RD results, then explain whether any difference changes the business decision.

### Regression Kink

Regression kink designs estimate whether the slope of an outcome changes when a
policy changes the marginal incentive rate. Unlike regression discontinuity,
the treatment does not jump from off to on. Instead, the intensity or payoff of
treatment changes at the threshold.

Commercial examples include commission plans where payout rate changes after a
quota threshold, discount schedules where marginal discount changes after a
spend tier, shipping fees that change slope after a basket value, or customer
success coverage that increases gradually with account size.

Students should look for a visible slope change, not just a level change. The
running variable must be smooth around the kink, and the threshold should not
induce severe manipulation. Kink designs are powerful when the policy truly
changes marginal incentives, but weak when the business rule is fuzzy.

### Instrumental Variables

Instrumental variables estimate the causal effect of a treatment when ordinary
regression is confounded by selection. The instrument must move the treatment,
but it must not affect the outcome except through that treatment. IV is often
taught as two-stage least squares: first predict treatment from the instrument,
then regress the outcome on predicted treatment.

Marketing examples include ad auction pacing shocks that move impressions,
inventory constraints that move exposure, weather shocks that move footfall,
sales-territory rules that move contact intensity, or platform delivery rules
that move treatment but are plausibly unrelated to customer intent.

The first question is relevance: does the instrument strongly move treatment?
The second is exclusion: can we defend that the instrument affects the outcome
only through treatment? Students should report first-stage strength, compare
naive and IV estimates, explain the population affected by the instrument, and
be honest when exclusion is more of an argument than a test.

### Weak-Instrument Diagnostics

Weak instruments create unstable IV estimates. If the instrument barely changes
treatment, the second-stage estimate can become noisy, biased, or wildly
sensitive to small modeling choices. A first-stage F-statistic is a common
starting diagnostic, but students should also inspect the practical size of the
first-stage effect.

In commercial settings, many tempting instruments are weak. A tiny auction
delivery rule, a minor weather fluctuation, or a barely enforced sales
territory rule may not move exposure enough to support a credible causal
estimate. A weak instrument does not become valid because it is convenient.

Students should treat weak-instrument diagnostics as a stop sign, not a
footnote. If the instrument is weak, the right output may be a rejected design
memo explaining why IV is not credible for this question.

### Variable-Selection Priors For IV

When analysts have many possible instruments, variable-selection priors provide
a structured way to encode skepticism. In a Bayesian IV workflow, stronger
candidate instruments can receive priors that allow inclusion, while weaker or
less credible candidates can receive priors that shrink them toward exclusion.

This can matter in marketing because instrument candidates often come from
messy operational systems: delivery rules, platform throttling, sales queues,
inventory constraints, payment failures, or eligibility logic. Some candidates
are likely more defensible than others.

The prior must be justified in business language. "This instrument is favored
because the platform pacing rule moved delivery opportunity before bidding" is
more useful than "the model selected it." Students should document why each
candidate is relevant, why exclusion is plausible or not, and how sensitive the
estimate is to prior choices.

### Inverse Propensity Weighting

Inverse propensity weighting estimates treatment effects from observational
data by reweighting units according to their probability of receiving
treatment. Units that received an unlikely treatment get more weight because
they represent similar units that were rarely observed in that treatment state.

In marketing, IPW is useful for retention offers, sales outreach, CRM
campaigns, coupon targeting, lifecycle emails, or lead routing when treatment
was assigned by a targeting model rather than randomized. The method can reduce
observed imbalance between treated and untreated groups.

The critical limitation is unobserved confounding. If the targeting team used
signals that are not in the dataset, weighting cannot fix that missing
information. Students should inspect propensity overlap, trim extreme weights,
check weighted covariate balance, and state that the result is conditional on
observed selection variables.

### ANCOVA For Nonequivalent Groups

ANCOVA compares treated and control groups after adjusting for baseline outcome
and covariates. It is useful when groups were not randomized but have a
pre-treatment measure that is strongly related to the post-treatment outcome.
The model asks whether treatment predicts post-period performance after
accounting for where each unit started.

Commercial examples include pilot programs where treated stores were chosen by
management, onboarding interventions assigned to a subset of customers, or
sales coaching rolled out to selected teams. Baseline performance often
explains much of the post-period variation, so ANCOVA can be more precise than
a raw pre/post comparison.

ANCOVA is not a magic fix for biased assignment. It adjusts for measured
baseline differences, not unmeasured reasons a group was selected. Students
should compare baseline distributions, include strong pre-treatment covariates,
and avoid using post-treatment variables as controls.

## How To Choose The Design

Ask these questions in order:

1. Was treatment randomized?
2. If not, was treatment assigned by time, geography, a threshold, or a
   targeting model?
3. Do we have untreated controls?
4. Do treated and control units look similar before treatment?
5. Is the causal estimate needed as a decision memo, an MMM calibration input,
   or a capstone artifact?

The answer usually points to a small set of candidate designs:

| Business constraint | First design to try | Backup design |
|---|---|---|
| National campaign, no untreated market | Interrupted time series | Piecewise ITS or marketing-specific lift test |
| One treated market, many untreated donors | Synthetic control | Comparative ITS |
| Several treated markets | Multi-cell geolift | Difference-in-differences |
| Regional rollout with controls | Difference-in-differences | Synthetic control for each treated region |
| Threshold-based promotion | Regression discontinuity | Donut RD or local sensitivity checks |
| Endogenous ad exposure | Instrumental variables | Sensitivity analysis with stronger caveats |
| Observational targeting | Inverse propensity weighting | Doubly robust model or matched comparison |

## Translating Outputs Into Marketing Evidence

Each design should produce an output that can be reused:

| Output | Use it for |
|---|---|
| Absolute lift with uncertainty | MMM calibration prior or experiment result table |
| Relative lift | Stakeholder-friendly campaign performance summary |
| Incremental revenue or orders | ROI and payback calculations |
| Geo-specific ATT | Rollout decision by region |
| Local threshold effect | Promotion policy decision near the cutoff |
| IV estimate | Directional evidence when ordinary regression is confounded |
| Weighted treatment effect | Observational campaign evidence with explicit selection caveats |

## Guardrails

- Do not choose a design because it produces the biggest lift.
- Do not treat synthetic control weights as stable if the pre-period fit is
  poor.
- Do not use difference-in-differences without checking pre-trends.
- Do not extrapolate regression discontinuity estimates far from the threshold.
- Do not use weak instruments as if they solved endogeneity.
- Do not use propensity weighting to fix unobserved confounding.

## References

- CausalPy examples index:
  https://causalpy.readthedocs.io/en/stable/notebooks/index.html
- CausalPy lift testing with interrupted time series:
  https://causalpy.readthedocs.io/en/stable/notebooks/its_lift_test.html
- CausalPy multi-cell geolift:
  https://causalpy.readthedocs.io/en/stable/notebooks/multi_cell_geolift.html
- CausalPy difference-in-differences:
  https://causalpy.readthedocs.io/en/stable/notebooks/did_skl.html
- CausalPy regression discontinuity:
  https://causalpy.readthedocs.io/en/stable/notebooks/rd_skl.html
- CausalPy instrumental variables:
  https://causalpy.readthedocs.io/en/stable/notebooks/iv_pymc.html
- CausalPy inverse propensity weighting:
  https://causalpy.readthedocs.io/en/stable/notebooks/inv_prop_pymc.html
