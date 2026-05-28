# Reading 1: What is Marketing Mix Modeling?

**Estimated time: ~20 minutes**

---

## The Measurement Challenge

Every marketer faces the same fundamental question: *"Which of my marketing investments are actually working?"*

This question has been asked since the early days of advertising. John Wanamaker's famous quote - "Half the money I spend on advertising is wasted; the trouble is I don't know which half" - captures the challenge perfectly. Marketing Mix Modeling (MMM) is one of the most powerful tools we have to answer it.

## A Brief History of MMM

- **1960s-1970s:** Econometric models first applied to marketing data by academics and CPG companies
- **1980s-1990s:** MMM becomes standard practice at large advertisers (P&G, Unilever, etc.) through agencies like Nielsen, IRI, and Analytic Partners
- **2000s-2010s:** Digital attribution models (last-click, multi-touch) emerge and temporarily overshadow MMM
- **2015-present:** Privacy changes (GDPR, iOS 14.5, cookie deprecation) make attribution models unreliable, triggering an MMM renaissance
- **2022-present:** Open-source MMM frameworks (Meta's Robyn, Google's Meridian, PyMC-Marketing) democratize access

## The Measurement Landscape

There are three primary approaches to measuring marketing effectiveness:

### 1. Marketing Mix Modeling (MMM)
- **What:** Statistical model relating marketing inputs (spend, GRPs, impressions) to business outcomes (sales, revenue) over time
- **Granularity:** Aggregate (weekly/monthly data)
- **Strengths:** Captures offline media, doesn't require user-level tracking, models carry-over and saturation effects
- **Limitations:** Requires 2+ years of data, can't measure individual touchpoints, results are correlational without experimental calibration

### 2. Multi-Touch Attribution (MTA)
- **What:** Assigns credit to individual touchpoints in a customer's digital journey
- **Granularity:** User-level
- **Strengths:** Granular, real-time, captures digital customer journeys
- **Limitations:** Only tracks digital channels, cookie/tracking dependent, doesn't capture incrementality, dying due to privacy changes

### 3. Experiments (Incrementality Testing)
- **What:** Randomized controlled tests (A/B tests, geo experiments) that establish causal impact
- **Granularity:** Varies (user-level for A/B, geo-level for lift tests)
- **Strengths:** Gold standard for causality, directly measures incrementality
- **Limitations:** Expensive, slow, can only test one thing at a time, requires holdout (lost revenue)

### Triangulation: The Best Practice

No single method is perfect. Best-in-class measurement programs combine all three:
- **MMM** provides the holistic, always-on view across all channels
- **Experiments** validate MMM results and establish causal ground truth
- **Attribution** provides tactical, real-time optimization signals for digital channels

This course focuses primarily on MMM, with coverage of attribution (Week 3) and experimentation (Week 4).

## Why MMM is Relevant Again

Several industry shifts have put MMM back at the center of measurement:

1. **iOS 14.5 (2021):** Apple's App Tracking Transparency reduced mobile attribution accuracy by 30-50%
2. **Cookie deprecation:** Third-party cookies are being phased out, undermining digital attribution
3. **Privacy regulations:** GDPR, CCPA, and similar laws restrict user-level tracking
4. **Open-source tools:** Meta, Google, and the PyMC community have released free, powerful MMM frameworks
5. **Cloud computing:** Bayesian methods that were computationally prohibitive are now feasible

## How MMM Works (Conceptual Overview)

At its core, MMM is a regression model:

```
Sales = Base + f(TV) + f(Digital) + f(Social) + ... + Controls + Error
```

But it's not just any regression. MMM incorporates two critical transformations:

1. **Adstock (carry-over):** Marketing effects don't stop when the ad stops. TV viewed this week still influences purchases next week. Adstock models this decay.

2. **Saturation (diminishing returns):** The 10th million dollars of TV spend doesn't generate the same return as the 1st million. Saturation curves capture this.

We'll build these from scratch in Week 2.

## What You'll Build in This Course

By the end of this bootcamp, you will have:
- Built an MMM from scratch using OLS and Bayesian methods
- Used Google's Meridian production framework
- Implemented Shapley and Markov attribution models
- Designed and analyzed a geo experiment
- Created a complete measurement plan

## Key Takeaways

1. MMM, attribution, and experiments are complementary - use all three (triangulation)
2. Privacy changes have made MMM more important than ever
3. Modern MMM is Bayesian, open-source, and accessible
4. The goal is not just measurement, but better marketing decisions

---

*Next reading: "Statistics Refresher for Marketing Science"*
