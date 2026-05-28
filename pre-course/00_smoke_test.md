# Environment Verification - Smoke Test

This notebook verifies that your Python environment is ready for the **June-26 Marketing Science Bootcamp**.

The default environment is intentionally lightweight. Core checks must pass for Week 0 and Week 1. Optional Bayesian, Meridian, GeoLift, and app checks are reported separately because those stacks are installed only when a later week needs them.

Run every cell from top to bottom. At the end you will see whether the core setup is ready and which optional extras are installed.


## 1. Python Version



```python
import sys
print(f"Python version: {sys.version}")

```

## 2. Package Import Checks

Core packages are required after `uv sync`. Optional packages are expected only after installing the matching extra, such as `uv sync --extra bayesian` or `uv sync --extra meridian`.



```python
# We will collect required and optional results separately
core_results = []
optional_results = []

def check_package(import_name, label=None, required=True, version_attr="__version__"):
    label = label or import_name
    target = core_results if required else optional_results
    try:
        module = __import__(import_name)
        version = getattr(module, version_attr, "OK")
        group = "CORE" if required else "OPTIONAL"
        print(f"PASS  -  {group:<8} {label} {version}")
        target.append((label, True, version))
    except ImportError as e:
        status = "FAIL" if required else "SKIP"
        group = "CORE" if required else "OPTIONAL"
        print(f"{status:<5} -  {group:<8} {label}: {e}")
        target.append((label, False, str(e)))

```


```python
check_package("pandas", "pandas", required=True)

```


```python
check_package("numpy", "numpy", required=True)

```


```python
check_package("matplotlib", "matplotlib", required=True)

```


```python
check_package("seaborn", "seaborn", required=True)

```


```python
check_package("scipy", "scipy", required=True)

```


```python
check_package("statsmodels", "statsmodels", required=True)

```


```python
check_package("sklearn", "sklearn", required=True)

```


```python
check_package("networkx", "networkx", required=True)

```


```python
check_package("openpyxl", "openpyxl", required=True)

```


```python
check_package("jupyter", "jupyter", required=True)

```

### Optional Package Checks

These are useful later in the course but are not required for the default Week 0 setup.



```python
check_package("pymc", "pymc", required=False)

```


```python
check_package("arviz", "arviz", required=False)

```


```python
check_package("pymc_marketing", "pymc_marketing", required=False)

```


```python
check_package("meridian", "google-meridian", required=False)

```


```python
check_package("rpy2", "rpy2", required=False)

```


```python
check_package("shiny", "shiny", required=False)

```

## 3. Data File Check

Verify that the workshop Excel file can be loaded.



```python
data_pass = False
try:
    import pandas as pd
    data = pd.read_excel("../data/MMM_Workshop_Data.xlsx", sheet_name="Data")
    print(f"PASS  -  Workshop data loaded successfully: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"         Columns: {list(data.columns)}")
    data_pass = True
except FileNotFoundError:
    print("FAIL  -  File not found: ../data/MMM_Workshop_Data.xlsx")
    print("         Make sure the data folder is in the expected location.")
except Exception as e:
    print(f"FAIL  -  Could not load data: {e}")

```

## 4. Summary



```python
import sys

print("=" * 60)
print("ENVIRONMENT VERIFICATION SUMMARY")
print("=" * 60)
print(f"\nPython version : {sys.version}")
print(f"Platform       : {sys.platform}\n")

print("-" * 60)
print(f"{'Core Package':<20} {'Status':<10} {'Version / Error'}")
print("-" * 60)

core_passed = True
for pkg, ok, info in core_results:
    status = "PASS" if ok else "FAIL"
    if not ok:
        core_passed = False
    print(f"{pkg:<20} {status:<10} {info}")

data_status = "PASS" if data_pass else "FAIL"
if not data_pass:
    core_passed = False
print(f"{'data file':<20} {data_status:<10} {'../data/MMM_Workshop_Data.xlsx'}")

print("\n" + "-" * 60)
print(f"{'Optional Package':<20} {'Status':<10} {'Version / Error'}")
print("-" * 60)
for pkg, ok, info in optional_results:
    status = "PASS" if ok else "SKIP"
    print(f"{pkg:<20} {status:<10} {info}")

print("-" * 60)
print()

if core_passed:
    print("\033[92m" + "=" * 60)
    print("  CORE CHECKS PASSED. Your Week 0/Week 1 environment is ready.")
    print("=" * 60 + "\033[0m")
    missing_optional = [pkg for pkg, ok, _ in optional_results if not ok]
    if missing_optional:
        print("\nOptional packages not installed:", ", ".join(missing_optional))
        print("Install the relevant extra only when that week needs it.")
else:
    failed = [pkg for pkg, ok, _ in core_results if not ok]
    if not data_pass:
        failed.append("data file")
    print("\033[91m" + "=" * 60)
    print("  WARNING: Core checks FAILED!")
    print(f"  Failed items: {', '.join(failed)}")
    print("  Please fix the core issues above before Session 1.")
    print("=" * 60 + "\033[0m")

```
