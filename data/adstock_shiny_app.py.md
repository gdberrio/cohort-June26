# data/adstock_shiny_app.py

## Purpose

Interactive Shiny for Python web application that lets students explore adstock and saturation transformations visually. Provides slider controls for transformation parameters and immediately renders updated decay curves and diminishing returns plots. Used as a supplemental learning tool in Week 2 when introducing adstock and saturation concepts.

## Dependencies

```
shiny, numpy, pandas, matplotlib, scipy (stats.weibull_min), nest_asyncio
```

## Running the App

```bash
# From the cohort-June26/ directory
python data/adstock_shiny_app.py
```

Opens a local web server (typically at `http://127.0.0.1:8000`).

## Module Sections

### 1. Transformation Functions (lines 19-127)

The app includes its own standalone implementations of the transformation functions (not imported from `utils/mmm_utils.py`) so it can run independently.

#### `adstock_geometric(x, theta) -> dict`

Geometric (Koyck) adstock: `x_decayed[t] = x[t] + theta * x_decayed[t-1]`.

- **Returns**: Dict with `x`, `x_decayed`, `thetaVecCum`, `inflation_total`.

#### `normalize(x) -> np.ndarray`

Min-max normalisation to [0, 1]. Used internally by Weibull adstock.

#### `adstock_weibull(x, shape, scale, windlen=None, adstockType="pdf") -> dict`

Weibull-distributed adstock transformation supporting both CDF and PDF variants.

- **adstockType**: `"cdf"` for monotone decay, `"pdf"` for flexible delayed-peak shapes.
- **Returns**: Dict with `x`, `x_decayed`, `thetaVecCum`, `inflation_total`, `x_imme`.

#### `saturation_power(x, n, theta) -> np.ndarray`

Combined power + geometric decay transformation: `x_sat[t] = x[t]^n + theta * x_sat[t-1]`.

#### `saturation_hill(x, alpha, inflexion) -> np.ndarray`

Hill function saturation: `x^alpha / (x^alpha + inflexion^alpha)`.

### 2. Sample Data (lines 129-166)

Hardcoded 32-week media spend series (`spends_data`) used as the demo input for all visualisations. Contains a realistic flighted TV spend pattern with zero-spend weeks, varying spend levels, and a large spike (week 12: $115K).

### 3. UI Layout (lines 168-211)

Built with `shiny.ui.page_fluid`:

- **Header**: Aryma Labs logo + title "Adstock Transformation in MMM"
- **Sidebar** (width 400px):
  - Adstock type selector: Geometric, Weibull CDF, Weibull PDF
  - Dynamic parameter sliders (change based on adstock type selection)
  - Saturation method selector: Hill or Power (Power only available with Geometric)
  - Dynamic saturation parameter sliders
- **Main panel**: Two tabbed plots via `navset_pill`:
  - "Decay Curve" — shows the cumulative decay weights over time
  - "Diminishing Returns Curve" — shows the saturation response curve

### 4. Server Logic (lines 214-336)

#### Reactive behaviour

- **Adstock type change**: Updates parameter sliders (theta for Geometric; shape/scale for Weibull CDF/PDF) and constrains saturation choices (Power is only available with Geometric).
- **Decay plot**: Computes decay curve from selected adstock function and parameters, renders as a line plot.
- **Saturation plot**: Applies the selected adstock to the spend data, then plots Hill or Power saturation curves over a simulated spend range (`0` to `1.2 * max_spend`).

#### Parameter ranges

| Adstock Type | Parameters | Slider Ranges |
|---|---|---|
| Geometric | theta | 0.00 - 1.00 (step 0.01) |
| Weibull CDF | shape, scale | shape: 0-2 (step 0.01), scale: 0-0.1 (step 0.01) |
| Weibull PDF | shape, scale | shape: 0-10 (step 0.1), scale: 0-0.1 (step 0.01) |

| Saturation Type | Parameters | Slider Ranges |
|---|---|---|
| Hill | alpha, gamma | alpha: 0.5-3.0 (step 0.1), gamma: 0.3-1.0 (step 0.1) |
| Power | n | 0.1-0.9 (step 0.1) |

## Usage Example

```bash
# Run directly:
python data/adstock_shiny_app.py

# Or from a notebook:
# (Open in a separate terminal as the app blocks the event loop)
```

Students adjust sliders to build intuition for:
- How theta controls the "memory" of geometric decay
- How Weibull shape/scale create flexible decay patterns
- How Hill alpha/gamma control the steepness and inflection of diminishing returns
- How Power n creates concave response curves
