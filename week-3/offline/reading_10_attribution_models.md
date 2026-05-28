# Reading 5: Introduction to Attribution Models

**Estimated time: ~30 minutes**

---

## The Attribution Problem

When a customer sees a TV ad on Monday, clicks a paid search ad on Wednesday, and converts through an email link on Friday, which channel gets credit for the sale? This is the **attribution problem** -- one of the most debated questions in marketing measurement.

Attribution is fundamentally about allocating credit for a conversion across the touchpoints that preceded it. It sounds simple, but in practice:

- Customers interact with **5-20+ touchpoints** before converting
- Channels influence each other (TV drives search, display builds awareness for later conversion)
- Some touchpoints are invisible (word-of-mouth, in-store exposure, organic recall)
- Correlation with conversion does not mean the touchpoint *caused* the conversion

Attribution models attempt to solve this credit-assignment problem. They range from simple rule-based heuristics to sophisticated data-driven algorithms.

---

## Rule-Based Attribution Models

Rule-based models apply predetermined rules to assign conversion credit. They are easy to implement but make arbitrary assumptions about how credit should be distributed.

### Model 1: Last-Click Attribution

The **last touchpoint** before conversion receives 100% of the credit.

```
Journey:  Display --> Social --> Paid Search --> [CONVERSION]

Credit:   Display: 0%    Social: 0%    Paid Search: 100%
```

**Why it's popular:** Simple, easy to implement, default in most analytics platforms (including Google Analytics historically).

**Why it's flawed:** Systematically over-credits lower-funnel channels (search, retargeting, email) and under-credits upper-funnel channels (TV, display, social). A customer who saw a TV ad, then a display ad, then searched -- the search gets all the credit, but it may not have happened without the earlier touchpoints.

### Model 2: First-Click Attribution

The **first touchpoint** in the journey receives 100% of the credit.

```
Journey:  Display --> Social --> Paid Search --> [CONVERSION]

Credit:   Display: 100%    Social: 0%    Paid Search: 0%
```

**Why it's used:** Emphasizes awareness and acquisition channels.

**Why it's flawed:** Ignores the nurturing and conversion steps entirely. A single display impression may have done very little to cause the final purchase.

### Model 3: Linear Attribution

Every touchpoint receives **equal credit**.

```
Journey:  Display --> Social --> Paid Search --> [CONVERSION]

Credit:   Display: 33.3%    Social: 33.3%    Paid Search: 33.3%
```

**Why it's used:** Feels "fair" -- acknowledges that all touchpoints contributed.

**Why it's flawed:** Treats a forgettable banner impression the same as a highly relevant search click. Not all touchpoints contribute equally.

### Model 4: Time-Decay Attribution

Touchpoints closer to conversion receive **more credit**, with credit decaying backward in time.

```
Journey:   Display --> Social --> Paid Search --> [CONVERSION]
Days out:    -14        -5          -1

Credit:    Display: 10%    Social: 30%    Paid Search: 60%
```

**Why it's used:** Reflects the intuition that recent interactions are more influential.

**Why it's flawed:** Still arbitrary -- the decay rate is chosen, not learned. Also biases toward lower-funnel channels, similar to last-click.

### Model 5: Position-Based (U-Shaped) Attribution

The **first and last** touchpoints each receive a large share (commonly 40%), with the remaining credit split among middle touchpoints.

```
Journey:  Display --> Social --> Email --> Paid Search --> [CONVERSION]

Credit:   Display: 40%    Social: 10%    Email: 10%    Paid Search: 40%
```

**Why it's used:** Balances awareness (first touch) and conversion (last touch) while giving some credit to the middle.

**Why it's flawed:** The 40/20/40 split is arbitrary. Why not 30/40/30? The model assumes the first and last touchpoints are always the most important, which is not always true.

### Summary of Rule-Based Models

| Model | Logic | Bias |
|---|---|---|
| Last-Click | 100% to last touch | Over-credits lower-funnel (search, retargeting) |
| First-Click | 100% to first touch | Over-credits upper-funnel (display, social) |
| Linear | Equal split | Treats all touches as equally important |
| Time-Decay | More credit to recent touches | Biases toward lower-funnel, arbitrary decay rate |
| Position-Based | 40/20/40 split (first/middle/last) | Arbitrary weights, assumes first and last always matter most |

---

## Why Rule-Based Models Are Fundamentally Flawed

Beyond the specific biases listed above, rule-based models share three deeper problems:

1. **Arbitrary rules, no learning:** The credit allocation is decided by the analyst, not derived from the data. There is no mechanism to discover whether a particular touchpoint actually influenced the outcome.

2. **No incrementality:** Attribution measures *who was present* in the journey, not *who made a difference*. A retargeting ad shown to someone already about to purchase gets full credit under last-click, even though removing it might not have changed the outcome.

3. **Ignores channel interactions:** Rule-based models treat each touchpoint independently. They cannot capture that TV + search together might be more effective than the sum of their individual effects.

---

## Data-Driven Attribution Models

Data-driven models learn credit allocation from the data rather than applying fixed rules. The two most established approaches are **Shapley Value attribution** and **Markov Chain attribution**.

### Shapley Value Attribution

Shapley values come from cooperative game theory, developed by Lloyd Shapley in 1953 (Nobel Prize in Economics, 2012). The core idea: when a group of "players" collaborate to produce an outcome, how do you fairly distribute the payoff?

**The Coalition Concept**

In attribution, the "players" are marketing channels and the "payoff" is conversions. The Shapley value calculates each channel's marginal contribution across *every possible coalition* (combination of channels).

Consider three channels: TV, Social, and Search. The possible coalitions are:

```
Coalition                  Conversions
-------                    -----------
{}                              0
{TV}                          100
{Social}                       80
{Search}                      120
{TV, Social}                  200
{TV, Search}                  250
{Social, Search}              220
{TV, Social, Search}          350
```

To compute TV's Shapley value, we look at TV's marginal contribution when added to each coalition that does not already include TV:

```
Adding TV to {}                 --> 100 - 0   = 100
Adding TV to {Social}           --> 200 - 80  = 120
Adding TV to {Search}           --> 250 - 120 = 130
Adding TV to {Social, Search}   --> 350 - 220 = 130

TV's Shapley value = average of marginal contributions
                   = (100 + 120 + 130 + 130) / 4
                   = 120
```

The same calculation is performed for Social and Search. The Shapley values will sum to the total conversions produced by the grand coalition (350).

**Properties of Shapley Values:**

| Property | Meaning |
|---|---|
| **Efficiency** | Credits sum to the total outcome |
| **Symmetry** | Channels with identical contributions receive equal credit |
| **Null Player** | A channel that adds nothing to any coalition receives zero credit |
| **Additivity** | The value for a combined game equals the sum of values for individual games |

**Limitations:**

- Computationally expensive: with *n* channels, there are 2^n coalitions. For 15 channels, that is 32,768 coalitions. Approximation methods (sampling) are used in practice.
- Assumes coalition values are observable, which requires enough data for each channel combination.
- Still measures association, not causation -- a channel may appear in many winning coalitions simply because it is always running, not because it is effective.

### Markov Chain Attribution

Markov Chain attribution models the customer journey as a sequence of states, where each state is a channel touchpoint, and transitions between states have probabilities.

**The Transition Matrix**

Given observed journeys, we construct a transition probability matrix:

```
              Start   Display  Social  Search  Convert  Drop
Start          --      0.40     0.35    0.15     0.00    0.10
Display        --       --      0.25    0.30     0.15    0.30
Social         --      0.10      --     0.35     0.20    0.35
Search         --      0.05     0.05     --      0.60    0.30
```

Read row-by-row: from the Start state, there is a 40% chance the customer's first touchpoint is Display, 35% chance it is Social, 15% chance it is Search, and 10% chance they drop off entirely.

**The Removal Effect**

The key concept in Markov attribution is the **removal effect**: what happens to the total conversion probability if we completely remove a channel from the graph?

1. Compute the **baseline conversion probability** using the full transition matrix
2. **Remove** a channel (replace all its transitions with absorption into the Drop state)
3. Compute the **new conversion probability** without that channel
4. The channel's **removal effect** = baseline conversion rate - conversion rate without the channel

```
Example:
  Baseline conversion probability:         32%
  Conversion probability without Display:  21%
  Conversion probability without Social:   18%
  Conversion probability without Search:   10%

  Removal effects:
    Display: 32% - 21% = 11%
    Social:  32% - 18% = 14%
    Search:  32% - 10% = 22%

  Normalized credit allocation:
    Display: 11/47 = 23.4%
    Social:  14/47 = 29.8%
    Search:  22/47 = 46.8%
```

**Advantages over rule-based:**
- Learns from actual journey data
- Accounts for channel position and sequence
- Captures how channels feed into each other

**Limitations:**
- Requires sufficient journey data (sparse paths are unreliable)
- Higher-order Markov chains (considering sequences of 2+ steps) are more accurate but require exponentially more data
- Still based on observed behavior, not causal experimentation

---

## Attribution vs Incrementality: The Critical Distinction

This is perhaps the most important conceptual distinction in marketing measurement:

| | Attribution | Incrementality |
|---|---|---|
| **Question** | "Who was in the room when the sale happened?" | "Who *caused* the sale to happen?" |
| **Method** | Observes customer journeys and assigns credit | Runs experiments to measure causal lift |
| **Counterfactual** | None -- only looks at what happened | Explicit -- compares treatment vs control |
| **Example** | Search gets 60% credit because it was the last click | A geo experiment shows turning off search reduces conversions by 25% |

**Why this matters:** Attribution can tell you that 80% of converters clicked a retargeting ad before purchasing. But incrementality testing might reveal that 90% of those people would have purchased anyway -- the retargeting ad had only a 10% incremental lift.

Attribution describes the **observed journey**. Incrementality measures the **causal lift** -- what would have happened in a world where the channel did not exist.

The gap between attributed and incremental credit can be enormous:
- **Branded search** often receives the most attribution credit but has low incrementality (people searching your brand name were likely going to buy anyway)
- **Upper-funnel channels** (TV, podcasts, OOH) often receive little attribution credit but may have high incrementality because they drive awareness that feeds the entire funnel

---

## How Attribution Fits Into the Measurement Trinity

As introduced in Reading 1, best-in-class measurement combines three approaches:

```
                    +---------------------+
                    |   Marketing Mix     |
                    |   Modeling (MMM)    |
                    |                     |
                    | Holistic, aggregate |
                    | All channels        |
                    | Strategic planning  |
                    +----------+----------+
                               |
              +----------------+----------------+
              |                                 |
   +----------+----------+           +----------+----------+
   |   Attribution        |           |   Experiments       |
   |                      |           |                     |
   | Journey-level        |           | Causal, gold        |
   | Digital channels     |           | standard            |
   | Tactical             |           | Validates MMM       |
   | optimization         |           | and attribution     |
   +----------------------+           +---------------------+
```

**How they complement each other:**

- **MMM** provides strategic, cross-channel budget allocation (including offline). It answers: "How should I allocate my $50M budget across channels next quarter?"
- **Attribution** provides tactical, real-time optimization within digital channels. It answers: "Which ad creative and targeting combination should I scale today?"
- **Experiments** provide causal ground truth and calibrate both MMM and attribution. They answer: "Is this channel actually *causing* incremental conversions?"

When all three agree, you have high confidence. When they disagree, it signals an opportunity to investigate further -- and experiments serve as the tiebreaker.

---

## Practice Questions

**Question 1:** A customer journey looks like this: Display Ad (Day 1) --> Facebook Ad (Day 5) --> Google Search (Day 12) --> Email (Day 14) --> Purchase. Under last-click attribution, email gets 100% credit. Under a position-based (40/20/40) model, how much credit does each touchpoint receive?

**Question 2:** You compute Shapley values for three channels and find TV = 45, Social = 30, Search = 25. A geo experiment reveals that turning off TV reduces conversions by only 10% (not 45%). What might explain this discrepancy, and what does it tell you about attribution vs incrementality?

**Question 3:** In a Markov Chain model, you find that removing the Display channel reduces total conversion probability from 30% to 28%, while removing Search reduces it from 30% to 12%. What does this tell you about the relative importance of these channels? Would you reallocate budget based solely on this analysis? Why or why not?

**Question 4:** Your CMO says: "Last-click attribution shows that branded search drives 60% of our conversions. We should increase search budget." Using concepts from this reading, construct a counter-argument.

**Question 5:** A colleague argues that data-driven attribution (Shapley or Markov) solves the problems of rule-based models. Do you agree? What fundamental limitation do even data-driven attribution models share with rule-based ones?

---

*Next: Session 5 will cover Google Meridian for production MMM*
