# YUVA Week 3 — Statistical Analysis and Predictive Modeling using R

## Project overview
This project uses R's built-in `mtcars` dataset (32 automobiles) to explore fuel efficiency (`mpg`), conduct hypothesis tests, fit a multiple linear regression model, validate predictions, and inspect model diagnostics.

## Dataset rationale
`mtcars` is a compact, reproducible dataset included with R. It contains fuel efficiency, vehicle weight, horsepower, transmission, and other automobile characteristics. The outcome `mpg` is numeric, making multiple linear regression appropriate. Its small sample size is an important limitation and is discussed in the report.

## Run in VS Code (PowerShell)
Open this project folder in VS Code, then run:
```powershell
& "C:\Program Files\R\R-4.6.2\bin\Rscript.exe" "scripts/statistical_modeling.R"
```
If R is installed elsewhere, replace the path with your Rscript.exe location. The script uses base R only and requires no package installation.

## Outputs
- `outputs/figures/`: exploratory plots and regression diagnostics
- `outputs/tables/`: summary statistics, test metrics, cross-validation metrics, coefficients, and predictions
- `report/`: project report

## Methods
- Pearson correlation test: `mpg` versus `wt`
- Welch two-sample t-test: `mpg` by transmission type
- Shapiro-Wilk normality checks
- 80/20 hold-out split
- Multiple linear regression: `mpg ~ wt + hp + am`
- 10-fold cross-validation on training data
- RMSE, MAE, test R-squared, residual and Q-Q plots

## Important interpretation note
The dataset is very small and observational. Associations do not establish causation. Cross-validation estimates are variable with such a small sample; results should be treated as a learning exercise, not a production-grade vehicle-efficiency predictor.
