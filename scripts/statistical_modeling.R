# YUVA Internship - Week 3
# Statistical Analysis and Predictive Modeling using R
# Dataset: built-in mtcars (32 automobiles; no external download required)

set.seed(2026)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)

# 1. Load and inspect data
data("mtcars")
df <- mtcars
df$am_label <- factor(df$am, levels = c(0, 1), labels = c("Automatic", "Manual"))
cat("\n--- DATA STRUCTURE ---\n")
str(df)
cat("\n--- SUMMARY STATISTICS ---\n")
print(summary(df))
write.csv(summary(df), "outputs/tables/summary_statistics.csv")

# 2. Exploratory analysis and hypotheses
cat("\n--- CORRELATION: MPG AND WEIGHT ---\n")
cor_test <- cor.test(df$mpg, df$wt, method = "pearson")
print(cor_test)

# H0: mean mpg is equal for automatic and manual transmission cars.
# H1: mean mpg differs between transmission groups.
cat("\n--- TWO-SAMPLE T TEST: MPG BY TRANSMISSION ---\n")
tt <- t.test(mpg ~ am_label, data = df)
print(tt)

# Normality test of the response (diagnostic context only; regression residuals
# are assessed after fitting).
cat("\n--- SHAPIRO-WILK TEST: MPG ---\n")
print(shapiro.test(df$mpg))

# 3. Train/test split (80/20), stratification not needed for regression.
set.seed(2026)
train_id <- sample(seq_len(nrow(df)), size = floor(0.8*nrow(df)))
train <- df[train_id, ]
test <- df[-train_id, ]

# Multiple linear regression chosen for interpretable continuous outcome.
# Predictors: weight, horsepower, and transmission.
model <- lm(mpg ~ wt + hp + am, data = train)
cat("\n--- TRAINING MODEL SUMMARY ---\n")
print(summary(model))

# 4. Hold-out test performance
pred <- predict(model, newdata = test)
actual <- test$mpg
rmse <- sqrt(mean((actual-pred)^2))
mae <- mean(abs(actual-pred))
r2_test <- 1 - sum((actual-pred)^2)/sum((actual-mean(actual))^2)
test_metrics <- data.frame(Metric=c("RMSE","MAE","Test_R_squared"),
                           Value=c(rmse,mae,r2_test))
print(test_metrics)
write.csv(test_metrics, "outputs/tables/test_metrics.csv", row.names=FALSE)

# 5. 10-fold cross-validation on the training set.
set.seed(2026)
k <- 10
fold_id <- sample(rep(1:k, length.out=nrow(train)))
cv_results <- data.frame(Fold=integer(), RMSE=numeric(), MAE=numeric())
for (fold in 1:k) {
  cv_train <- train[fold_id != fold, ]
  cv_valid <- train[fold_id == fold, ]
  cv_model <- lm(mpg ~ wt + hp + am, data=cv_train)
  cv_pred <- predict(cv_model, newdata=cv_valid)
  cv_results <- rbind(cv_results,
    data.frame(Fold=fold,
      RMSE=sqrt(mean((cv_valid$mpg-cv_pred)^2)),
      MAE=mean(abs(cv_valid$mpg-cv_pred))))
}
cat("\n--- 10-FOLD CV RESULTS ---\n")
print(cv_results)
cat("Mean CV RMSE:", mean(cv_results$RMSE), "\n")
cat("Mean CV MAE:", mean(cv_results$MAE), "\n")
write.csv(cv_results, "outputs/tables/cross_validation_metrics.csv", row.names=FALSE)

# 6. Diagnostics on training model
res <- residuals(model)
cat("\n--- SHAPIRO-WILK TEST: MODEL RESIDUALS ---\n")
print(shapiro.test(res))
cat("\n--- BREUSCH-PAGAN-LIKE AUXILIARY TEST ---\n")
# Base-R auxiliary regression: squared residuals against fitted values.
bp_aux <- lm(I(res^2) ~ fitted(model))
bp_stat <- nobs(model) * summary(bp_aux)$r.squared
bp_p <- pchisq(bp_stat, df=1, lower.tail=FALSE)
print(data.frame(BP_statistic=bp_stat, p_value=bp_p))
write.csv(data.frame(BP_statistic=bp_stat,p_value=bp_p),
          "outputs/tables/heteroskedasticity_test.csv", row.names=FALSE)

# 7. Visualizations / diagnostics
png("outputs/figures/01_mpg_vs_weight.png", width=1200, height=900, res=150)
plot(df$wt, df$mpg, pch=19, col="steelblue",
     xlab="Weight (1000 lbs)", ylab="Miles per gallon",
     main="Fuel efficiency versus vehicle weight")
abline(lm(mpg ~ wt, data=df), col="firebrick", lwd=2)
dev.off()

png("outputs/figures/02_mpg_distribution.png", width=1200, height=900, res=150)
hist(df$mpg, breaks=8, col="skyblue", border="white",
     main="Distribution of fuel efficiency", xlab="Miles per gallon")
dev.off()

png("outputs/figures/03_residuals_fitted.png", width=1200, height=900, res=150)
plot(fitted(model), res, pch=19, col="steelblue",
     xlab="Fitted values", ylab="Residuals",
     main="Residuals versus fitted values")
abline(h=0, col="firebrick", lwd=2)
dev.off()

png("outputs/figures/04_qq_plot.png", width=1200, height=900, res=150)
qqnorm(res, pch=19, col="steelblue", main="Normal Q-Q plot of residuals")
qqline(res, col="firebrick", lwd=2)
dev.off()

png("outputs/figures/05_actual_predicted.png", width=1200, height=900, res=150)
plot(actual, pred, pch=19, col="steelblue",
     xlab="Actual MPG", ylab="Predicted MPG",
     main="Hold-out: actual versus predicted")
abline(0,1,col="firebrick",lwd=2)
dev.off()

# 8. Export coefficients and predictions
write.csv(as.data.frame(summary(model)$coefficients),
          "outputs/tables/model_coefficients.csv")
prediction_output <- data.frame(Actual=actual, Predicted=as.numeric(pred),
                                 Error=actual-as.numeric(pred))
write.csv(prediction_output, "outputs/tables/test_predictions.csv", row.names=FALSE)

cat("\nAnalysis complete. Check outputs/figures and outputs/tables.\n")
