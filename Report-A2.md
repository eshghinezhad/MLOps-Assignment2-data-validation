# Assignment 2 Report – Data Validation & Testing
**Course:** MAI201 MLOps  
**Student:** Monireh Eshghinezhad  
**Date:** 2026-06-22

---

## 1. Great Expectations Validation Results

> **Screenshot instructions:** Run all cells in `assignment2.ipynb`, then take a
> screenshot of the printed `summary_df` table (Part 3, "Print a per-expectation
> summary table" cell) and paste it here.  
> Also open `gx/uncommitted/data_docs/local_site/index.html` in your browser and
> take a screenshot of the Great Expectations Data Docs page.

*(Replace this line with your screenshot of the GE validation results table.)*

*(Replace this line with your screenshot of the GE Data Docs HTML page.)*

---

## 2. Data Quality Issues Found

All counts are produced by the pandas analysis cells in Part 3 of `assignment2.ipynb`.
Run those cells to get the exact numbers from your version of the dataset.

| # | Issue | Count* | Business Impact |
|---|-------|--------|-----------------|
| 1 | Null `customer_id` | see notebook | Record cannot be identified or joined to other tables |
| 2 | Duplicate `customer_id` | see notebook | Over-represents those customers during model training |
| 3 | Null `age` | see notebook | Missing feature — requires imputation or row removal |
| 4 | Out-of-range `age` (< 0 or > 120) | see notebook | Biologically impossible; corrupts age-based model features |
| 5 | Null `email` | see notebook | Cannot contact or join customer records on email |
| 6 | Invalid `email` format (fails regex) | see notebook | Unusable for communication or lookup matching |
| 7 | Null `salary` | see notebook | Missing numeric feature — imputation required |
| 8 | Negative `salary` | see notebook | Data entry error; distorts all numeric salary-based features |
| 9 | `country` not in {USA, Canada, UK, Australia} | see notebook | Records from unsupported markets included in training |
| 10 | Unparseable `signup_date` | see notebook | Time-series features (tenure, churn windows) will fail |

\* Fill in counts from the "DATA QUALITY ISSUES SUMMARY" table printed at the end of
Part 3 when you run the notebook.

---

## 3. pytest Execution Screenshot

> Run the following command in your terminal from the project root:
> ```bash
> pytest test_data_utils.py -v --tb=short
> ```
> Take a screenshot of the terminal output showing all tests passing (green dots / PASSED)
> and paste it here.

*(Replace this line with your screenshot of the pytest terminal output.)*

Expected output summary:
```
test_data_utils.py::TestLoadCSV::test_file_not_found_raises          PASSED
test_data_utils.py::TestLoadCSV::test_empty_file_raises              PASSED
test_data_utils.py::TestLoadCSV::test_successful_load_returns_dataframe PASSED
test_data_utils.py::TestLoadCSV::test_successful_load_has_expected_columns PASSED
test_data_utils.py::TestCleanPhone::test_dashes_removed              PASSED
test_data_utils.py::TestCleanPhone::test_dots_removed                PASSED
test_data_utils.py::TestCleanPhone::test_spaces_removed              PASSED
test_data_utils.py::TestCleanPhone::test_parentheses_removed         PASSED
test_data_utils.py::TestCleanPhone::test_plain_digits_unchanged      PASSED
test_data_utils.py::TestCleanPhone::test_none_returns_empty          PASSED
test_data_utils.py::TestCleanPhone::test_nan_string_returns_empty    PASSED
test_data_utils.py::TestCleanPhone::test_too_short_returns_empty     PASSED
test_data_utils.py::TestCleanPhone::test_negative_placeholder_returns_empty PASSED
test_data_utils.py::TestCleanPhone::test_mixed_format                PASSED
test_data_utils.py::TestValidateEmail::test_standard_email           PASSED
test_data_utils.py::TestValidateEmail::test_subdomain_email          PASSED
test_data_utils.py::TestValidateEmail::test_plus_tag_email           PASSED
test_data_utils.py::TestValidateEmail::test_numeric_local_part       PASSED
test_data_utils.py::TestValidateEmail::test_missing_at_sign          PASSED
test_data_utils.py::TestValidateEmail::test_missing_domain           PASSED
test_data_utils.py::TestValidateEmail::test_missing_tld              PASSED
test_data_utils.py::TestValidateEmail::test_starts_with_at           PASSED
test_data_utils.py::TestValidateEmail::test_invalid_string           PASSED
test_data_utils.py::TestValidateEmail::test_none_returns_false       PASSED
test_data_utils.py::TestValidateEmail::test_empty_string_returns_false PASSED
test_data_utils.py::TestValidateEmail::test_whitespace_only_returns_false PASSED
test_data_utils.py::TestValidateEmail::test_nan_string_returns_false PASSED

27 passed in X.XXs
```

---

## 4. Reflection – Which Data Quality Issue Would Most Impact ML Model Performance?

**Negative and missing salary values** would most severely impact ML model performance.

`salary` is a continuous numeric feature used directly in model arithmetic — distances
in k-NN, split thresholds in decision trees, and linear combinations in regression.
Negative salaries are impossible in reality, yet the model treats them as valid large
negative numbers. This causes three compounding problems:

1. **Distorted feature distribution** — Min-max scaling compresses all real salaries
   into a narrow range while the invalid negatives dominate the lower bound of the
   scale, making the feature almost constant for valid records.

2. **False signal learned by the model** — The model may learn a spurious relationship
   between negative salary and the target variable, when negative salary only predicts
   a data entry mistake.

3. **Cascading imputation error** — If missing salaries are filled with the column
   mean, that mean is itself corrupted by the negatives, spreading the error to every
   imputed row.

By contrast, issues like inconsistent phone formats affect a column rarely used as a
model feature, and country outliers are handled safely by one-hot encoding (an
"unsupported country" category becomes its own binary feature).

**Recommended remediation:**
1. Replace negative salary values with `NaN`.
2. Impute with the **median** (robust to remaining outliers) after removing the corrupt rows.
3. Add a Great Expectations check `expect_column_values_to_be_between(min_value=0)` to
   prevent negatives from re-entering the pipeline.

---

## 5. File Index

| File | Purpose |
|------|---------|
| `assignment2.ipynb` | Main notebook — all five assignment parts with explanations |
| `data_utils.py` | Utility functions: `load_csv`, `clean_phone`, `validate_email` |
| `test_data_utils.py` | pytest unit tests for all three utility functions (27 tests) |
| `data_quality_report.html` | Standalone HTML data quality report |
| `gx/` | Great Expectations project (suite JSON + interactive HTML docs) |
| `data/customer_data.csv` | Original messy dataset (unmodified) |
