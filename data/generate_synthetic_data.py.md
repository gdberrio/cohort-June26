# data/generate_synthetic_data.py

## Purpose

Generates the two synthetic datasets used across the bootcamp curriculum. Designed to be run once during setup or whenever fresh data is needed. Both datasets embed realistic marketing patterns that make pedagogical exercises meaningful.

## Dependencies

```
numpy, pandas, os
```

## Usage

```bash
python data/generate_synthetic_data.py
```

**Outputs** (written to `data/`):
- `synthetic_multi_geo_data.csv` — 520 rows (5 geos x 104 weeks)
- `synthetic_journeys.csv` — 10,000 customer journeys

## Module Sections

### 1. Helper Functions (lines 28-72)

#### `apply_adstock(series, theta) -> np.ndarray`

Applies geometric (Koyck) adstock transformation to a 1-D spend array. Used internally by the geo data generator to create realistic media carry-over effects.

- **series**: Raw weekly spend values.
- **theta**: Decay rate in [0, 1].

#### `diminishing_returns(x, half_saturation=1.0) -> np.ndarray`

Hill-function saturation curve: `x / (x + half_saturation)`. Models diminishing returns on media spend.

- **half_saturation**: Input level at which output reaches 50% of maximum.

### 2. Dataset 1 — Multi-Geo Marketing Data (lines 75-256)

#### `generate_multi_geo_data(seed=42) -> pd.DataFrame`

Generates a synthetic multi-geo weekly marketing dataset mimicking what a brand would feed into Google Meridian or a Bayesian MMM framework.

**Structure**: 5 US geographic regions x 104 weeks = 520 rows.

| Column | Description |
|---|---|
| `date` | Weekly date (Monday, starting 2022-01-03) |
| `geo` | Geographic region: northeast, southeast, midwest, west, southwest |
| `revenue` | Weekly revenue in thousands of dollars |
| `tv_spend` | TV spend — flighted (~60% of weeks active, heavier in Q4) |
| `digital_spend` | Digital spend — always-on with weekly variation |
| `social_spend` | Social spend — moderate frequency (~75% active) |
| `search_spend` | Search spend — always-on, seasonally correlated |
| `display_spend` | Display spend — runs ~50% of weeks |
| `price_index` | Price index with slow drift (centered at 100) |
| `competitor_spend` | Competitor spend with upward trend |

**Revenue is driven by**:
- Geo-specific base levels (e.g., west = $900K/week, southwest = $550K/week)
- Seasonal patterns: sinusoidal annual cycle + Q4 holiday bump
- Media effects with channel-specific adstock decay and Hill saturation
- Price index (negative effect — higher prices reduce revenue)
- Competitor spend (negative effect)
- Random noise (~4% of base)

**Built-in ground truth** (useful for validating student models):

| Channel | Adstock Theta | Half-Saturation | Revenue Coefficient |
|---|---|---|---|
| TV | 0.70 | 200 | 0.30 |
| Digital | 0.10 | 120 | 0.20 |
| Social | 0.30 | 80 | 0.15 |
| Search | 0.15 | 100 | 0.25 |
| Display | 0.25 | 60 | 0.10 |

**Geo spend scaling**: northeast 1.10x, southeast 0.95x, midwest 0.80x, west 1.20x, southwest 0.70x.

### 3. Dataset 2 — Customer Journey Data (lines 259-383)

#### `generate_journey_data(n_journeys=10_000, seed=42) -> pd.DataFrame`

Generates synthetic multi-touch customer journey data for attribution modeling exercises (Shapley, Markov, heuristic models).

| Column | Description |
|---|---|
| `journey_id` | Integer ID (1 to n_journeys) |
| `touchpoint_sequence` | Semicolon-separated channel names (e.g., `"Organic Search;Email;Direct"`) |
| `num_touchpoints` | Length of the sequence (geometric distribution, mostly 1-6) |
| `converted` | Binary (1 = conversion, 0 = no conversion) |
| `revenue` | Log-normal revenue for converted journeys (~$50 mean); 0 for non-converted |

**Channels**: Paid Search, Display, Social Media, Email, Organic Search, Direct.

**Behavioural patterns embedded**:
- Position-dependent channel probabilities (e.g., Organic Search common as first touch; Direct/Paid Search common as last touch)
- Conversion probability via logistic model: base logit + touchpoint count + channel-level lift + diversity bonus
- Average conversion rate tuned to ~12%

**Channel conversion lift** (additive to logit):

| Channel | Lift |
|---|---|
| Direct | +0.40 |
| Email | +0.30 |
| Paid Search | +0.25 |
| Organic Search | +0.15 |
| Social Media | +0.05 |
| Display | -0.10 |

### 4. Main / CLI (lines 386-463)

#### `print_summary(df, name) -> None`

Prints descriptive summary statistics for a generated dataset, including revenue by geo, spend averages, conversion rates, and touchpoint distributions.

**When run as a script** (`__main__`):
1. Generates both datasets with `seed=42`
2. Saves CSVs to the same directory as the script
3. Prints comprehensive summaries to stdout
