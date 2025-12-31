#@# FINAL PROJECT TEMPLATE
#@#
#@# Submission of LLM-generated code will result in a zero grade.
#@# Use mainly the techniques you learned in class. If you go out of that set, explain why.
#@# You are not permitted to collaborate on this assignment.
#@#
#@# Instructions:
#@# 1. Do not modify any line beginning with #@#
#@# 2. Do not begin any line in your solution programs or comments with #@#
#@# 3. Paste your R code below the line PASTE R CODE BELOW HERE for each answer.
#@# 4. Ensure each answer corresponds to the correct question number.
#@# 5. Provide code for all 5 questions, even if incomplete, to avoid parsing errors.
#@# 6. Save this file as a plain .txt file and upload via the Google Form.
#@#=================================
#@# --START OF QUESTION_1--
#@# Title: Missing Data Analysis and Treatment
#@# A city measures 9 variables X1-X8 and Y every day. Some of the measurements are missing. Describe the situation and address it. Justify what you do.
#@# --END OF QUESTION_1--
#@# --START OF ANSWER_1--
#@# PASTE R CODE BELOW HERE

library(tidyverse)
library(tidymodels)
library(dplyr)
  
getwd()
  
city <- read.csv("data.csv")
head(city)
dim(city)
#checking for missing values
sum(is.na(city))    
colSums(is.na(city))
str(city)
summary(city)
  
#ok X1 and X2 are categorical need to be factored, and X3 to X8 are numerical(maybe like distance or something)
#Y has no missing values so we don't need to do anything
#checking proportions of X1 and X2
X1_prop <- table(city$X1, useNA = "ifany")
X1_prop
  
X2_prop <- table(city$X2, useNA = "ifany")
X2_prop

#we will use dummy variable  later  
#let us factor the nominal predictors 
  
city$X1 <- factor(city$X1, levels = c("Yes", "No"))
city$X2 <- factor(city$X2, levels = c( "Cold","Neutral", "Hot"))
  
str(city)
#I think the bes method would be replace missing values with mode, as mean and median won't make sense
#what if we delete all rows which have NA values?
#let's check number of rows which have atleast 1 missing values

rows_missing_counts <- rowSums(is.na(city))
rows_with_missing_values <- sum(rows_missing_counts > 0)
table(rows_with_missing_values)
  
#Nope, thats 35% of our dataset we can't delete them
#dropping them would shrink data and risk bias.
  
#lets visualize data to median or mean 
city %>%
  select(X3:X8) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  facet_wrap(~ Variable, scales = "free") +
  theme_minimal()
  
#checking skewness and kurtosis
library(e1071)
  
city %>%
  summarise(across(X3:X8, ~skewness(.x, na.rm = TRUE))) 
  
city %>%
  summarise(across(X3:X8, ~kurtosis(.x, na.rm = TRUE)))
  
#X5 and X6 missing values have to be replaced with median and for rest numeric predictors with mean 
#reason being for |skewness| ≤ 0.5 , mean is reasonable, otherwise median is better and X5 and X6 have high skewness with heavy tails.
#X1 and X2 missing values have to be replaced with mode
#even though yes and no values are almost similar I am going with mode as there won't be such big class imbalance by using mode.

#now lets write a recipe for missing vales 
  
rec_q1 <- recipe(Y ~ ., data = city) %>%
  step_impute_mode(all_nominal_predictors()) %>%                   
  step_impute_mean(any_of(c("X3", "X4", "X7", "X8"))) %>%          
  step_impute_median(any_of(c("X5", "X6"))) %>%                    
  step_zv(all_predictors())
  
#prep and bake 
rec_q1_prep <- prep(rec_q1, training = city, verbose = TRUE)
city_missing   <- bake(rec_q1_prep, new_data = city)

#checking if there are any missing value left
colSums(is.na(city_missing)) #nope no missing values left 

dim(city_missing)# and we did not lose any data.

#next step would be to transform the data as we can see some heavy skewness in X5 and X6
#X5 just needs normal log transformation as it is right skewed 
#but X6 is left skewed so we have to first reflect it and then use log transformation

city_transformed <- city_missing %>%
  mutate(
    X5_log = log1p(X5),
    X6_reflected = max(X6, na.rm = TRUE) + 1 - X6,
      X6_log = log1p(X6_reflected)
    ) %>%
    select(-X6_reflected)  
  
# Check improvement
city_transformed %>%
  summarise(
      X5_log_skew = skewness(X5_log),
      X5_log_kurt = kurtosis(X5_log),
      X6_log_skew = skewness(X6_log),
      X6_log_kurt = kurtosis(X6_log)
    )
  
city_transformed
  
#I want to check before v/s after tranfrmation 

numeric_before <- city %>%
  select(X3, X4, X5, X6, X7, X8) %>%
  mutate(stage = "Before")

numeric_after <- city_transformed %>%
  select(X3, X4, X5 = X5_log, X6 = X6_log, X7, X8) %>%
  mutate(stage = "After")


numeric_long <- bind_rows(numeric_before, numeric_after) %>%
  pivot_longer(cols = -stage, names_to = "Variable", values_to = "Value")


ggplot(numeric_long, aes(x = Value)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  facet_grid(stage ~ Variable, scales = "free") +
  theme_minimal() +
  labs(title = "Numeric Predictors Before vs After Transformation",
       x = "Value",
       y = "Count")

#yup transformations worked, the data looks good
#it might look like transformaton did not work on X5 and X6, but it is becuase of scale.
#the values are low because of transformation and on that scale it looks like they are squished to corner
#but transformation was succesfull as it was confirmed by skewness and kurtosis value. 

#let us use this data as final data 
city_final <- city_transformed

#@# --END OF ANSWER_1--
#@#=================================
#@# --START OF QUESTION_2--
#@# Title: Data Visualization and Distribution Analysis
#@# Use visualizations to understand the distributions of variables and their relationship to each other. Describe what you observe. Justify what you do.
#@# --END OF QUESTION_2--
#@# --START OF ANSWER_2--
#@# PASTE R CODE BELOW HERE

city_plot <- city_final %>%
  mutate(
    X5 = X5_log,
    X6 = X6_log
  ) %>%
  select(-X5_log, -X6_log)

#new df just for easy usage

head(city_plot)
head(city_transformed)
#yup, correct values at the correct columns, let us just cross veify skewness and kurtosis 

city_plot %>%
  summarise(across(X3:X8, ~skewness(.x, na.rm = TRUE))) 

city_plot %>%
  summarise(across(X3:X8, ~kurtosis(.x, na.rm = TRUE)))

#yup values look good

city_plot %>%
  pivot_longer(cols = c(X1, X2), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value, fill = Value)) +
  geom_bar() +
  facet_wrap(~ Variable, scales = "free") +
  theme_minimal() +
  labs(
    title = "Categorical Variable Distributions",
    x = "",
    y = "Count"
  )

#X1 shows a slightly higher proportion of “Yes” responses than “No”
#X2 has 3 categories and Nuetral is the most frequent

#visualization for numeric predictors

city_plot %>%
  pivot_longer(cols = X3:X8, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  facet_wrap(~ Variable, scales = "free") +
  theme_minimal() +
  labs(
    title = "Numeric Variable Distributions (After Transformation)",
    x = "Value",
    y = "Count"
  )

#yup, graph also looks good, with all of them being mostly symmetrical.
#Skewness values for all variables are close to zero, and kurtosis is within a mild range (no heavy tails)
#The transformations have successfully normalised X5 (originally right-skewed) and X6 (originally left-skewed).

#let us try correlation heatmap for X3–X8

corr_tbl <- city_plot %>% select(X3:X8) %>% cor()
as_tibble(corr_tbl, rownames = "Var1") %>%
  pivot_longer(-Var1, names_to = "Var2", values_to = "r") %>%
  ggplot(aes(Var1, Var2, fill = r)) +
  geom_tile() +
  scale_fill_gradient2(limits = c(-1,1)) +
  coord_equal() +
  theme_minimal() +
  labs(title = "Correlation Heatmap (X3–X8)", x = "", y = "", fill = "r")

#it is very clear about which pairs might be corelated 
#this shows strong positive correlations between pair  X3 and X4 and pair X7 and X8
#Other correlations are weak, indicating most variables carry independent information

#now let us check check predictors vs outcome 

city_plot %>%
  select(Y, X3:X8) %>%
  pivot_longer(-Y, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(Value, Y)) +
  geom_point(alpha = 0.25) +
  geom_smooth(se = FALSE, method = "lm", color = "red") +
  facet_wrap(~ Variable, scales = "free_x") +
  theme_minimal() +
  labs(title = "Predictors vs Y (Linear Fit)", x = "Predictor", y = "Y")


#The fitted regression lines are nearly flat, indicating that none of the predictors has a strong linear relationship with Y. 
#This means no single variable is likely to dominate the model, which is favourable for multivariate modelling

#let us confirm by finding correlation with Y
cors_to_Y <- city_plot %>%
  select(Y, X3:X8) %>%
  summarise(across(-Y, ~cor(.x, Y))) %>%
  pivot_longer(everything(), names_to = "Predictor", values_to = "r") %>%
  arrange(desc(abs(r)))
cors_to_Y

#this matches the graph we saw a above
#This is actually favourable for multivariate modelling where no single predictor dominates.

#now let's check group effects , I used to do this in my internship

# By X1
city_plot %>%
  pivot_longer(X3:X8, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(X1, Value, fill = X1)) +
  geom_boxplot(outlier.alpha = 0.2) +
  facet_wrap(~ Variable, scales = "free_y") +
  theme_minimal() +
  labs(title = "Predictors by X1", x = "X1", y = "Value")

# By X2
city_plot %>%
  pivot_longer(X3:X8, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(X2, Value, fill = X2)) +
  geom_boxplot(outlier.alpha = 0.2) +
  facet_wrap(~ Variable, scales = "free_y") +
  theme_minimal() +
  labs(title = "Predictors by X2", x = "X2", y = "Value")

#man this is kinda tough 
#according to me both categorical variables likely do not drive large differences in these numeric predictors on their own
#any predictive power from X1 or X2 is likely through interaction effects or in combination with the other predictors
#so maybe dummy variable is needed for X2
#lets check it in next ques with tests.

#let us draw q-q plots 
qq_df <- tidyr::pivot_longer(
  city_plot[, paste0("X", 3:8)],
  cols = everything(),
  names_to = "Variable",
  values_to = "Value"
)
ggplot(qq_df, aes(sample = Value)) + stat_qq() + stat_qq_line() +
  facet_wrap(~ Variable, scales = "free") + theme_minimal() +
  labs(title = "QQ-Plots for X3–X8")

#and yup everything deviated from normality, lets check this by shapiro test in next ques

#@# --END OF ANSWER_2--
#@#=================================
#@# --START OF QUESTION_3--
#@# Title: Statistical Testing for Variable Relationships
#@# If any data columns appear unusual, or very closely related, test your suspicions using statistics. Describe what you observe. Justify what you do.
#@# --END OF QUESTION_3--
#@# --START OF ANSWER_3--
#@# PASTE R CODE BELOW HERE

#Suspicion 1 – Multicollinearity (from above correlation)
library(car)
model_vif <- lm(Y ~ X3 + X4 + X5 + X6 + X7 + X8, data = city_plot)
vif(model_vif)
#the vif score is high(greater than 5) for X3,X4 ,X7 and X8 confirming our suspicions.
#This confirms high multicollinearity between these pairs, likely X3↔X4 and X7↔X8

#Suspicion 2 – Normality of numeric predictors(to even know which tests to use)
city_plot %>%
  select(X3:X8) %>%
  summarise(across(everything(), ~shapiro.test(.x)$p.value))
#as all values are below 0.05, we reject H0 which is "distribution is normal"
#so parametric tests cannot be used.


#Suspicion 3 – Independence between X1 and X2(was not completely clear above)
#we use chi-square test of independence 
tab_X1_X2 <- table(city_plot$X1, city_plot$X2)
chi_X1_X2 <- chisq.test(tab_X1_X2)
chi_X1_X2

#as p is greater than 0.05 , we fail to reject the null hypothesis that X1 and X2 are independent.
#So there is no statistical evidence that X1 and X2 are associated.
library(rcompanion)
cramerV(tab_X1_X2, bias.correct = TRUE)
#It also confrims our test.

#Suspicion 4 – Relationship between categorical and numeric predictors(was't completely sure about this, but reults were satisfying)

#now lets do  Wilcoxon rank-sum for X1 across each numeric predictor(cannot use paraetric tests as data is not normal)
wilcox_X1 <- map_dfr(paste0("X", 3:8), function(v) {
  wt <- wilcox.test(reformulate("X1", v), data = city_plot, exact = FALSE)
  tibble(Predictor = v, p_value = wt$p.value)
})
wilcox_X1

#took help from gpt to get all together, instead of doing individually

#as all p-values are greater than 0.05 we fail to reject null hypothesis
#(H₀: The distributions of the numeric variable are the same for the two groups of X1 ) 
# there’s no strong evidence that the medians differ between X1 groups.
# X1 may remain useful in the model, but not because it changes X3–X8’s distributions strongly.

#I want to do the same test for X2 where it has 3 categories ,the method to use is ANOVA. 
#but for that we need normal distribution, and I couldn't find any non-parametric ANOVA test from class slides.
#so I am using kruskal-Wallis test. (google) which is non-parametric ANOVA test.(in smple terms)

kruskal_X2 <- map_dfr(paste0("X", 3:8), function(v) {
  kt <- kruskal.test(reformulate("X2", v), data = city_plot)
  tibble(Predictor = v, p_value = kt$p.value)
})
kruskal_X2
#here the null hypothesis is that the distribution of the numeric variable is the same across all X2 groups.
#as all p-values are greater than 0.05 we fail to reject H0, that means 
#No evidence that X2 has a meaningful main effect on any numeric predictor.
#Any predictive influence from X2 will likely come through interactions with other variables rather than direct differences in means/medians.

#all my suspicions are cleared and we can go to question 4


#@# --END OF ANSWER_3--
#@#=================================
#@# --START OF QUESTION_4--
#@# Title: Predictive Modeling for Critical Condition Detection
#@# The city considers Y>112 to be a critical condition. 
#@# Develop two models that predicts if Y>112. 
#@# At least one of the models should be of a type that is interpretable. 
#@# At least one of the models should be of a type that is tunable. 
#@# Evaluate the performance of the two models. 
#@# Explore and describe performance tradeoffs with respect to threshold choice. 
#@# Explore and describe over/under fitting tradeoffs with respect to hyperparameter value.
#@# Describe what you observe. Justify what you do.
#@# --END OF QUESTION_4--
#@# --START OF ANSWER_4--
#@# PASTE R CODE BELOW HERE

#for this my plan is to use logistic regression and randomn forest models.
#I woud have to use dummy variables for X2(as there are 3 categories)
#target is Y>112, so it is basically a classification problem.
#so first I want to create a new column of yes and no where Y>112.
set.seed(123)

city_critical <- city_plot %>%
  mutate(
    critical = factor(ifelse(Y > 112, "yes", "no"),
                      levels = c("no", "yes"))
  ) %>%
  select(-Y) 
#took out Y as it is now a classification problem with critical as output, and Y is not needed.

head(city_critical)

#let us get the proportions of yes and no
table(city_critical$critical) %>% prop.table()
#A naive model which predics all no will have 78% accuracy
#there is a class imbalance, we can either populate the minority or remove the majority
#as we have less data it is better to populate the minority(damn we are using all the concepts taught in class in this assignment)

#first let us split data 
split <- initial_split(city_critical, prop = 0.8, strata = critical)
city_train  <- training(split)
city_test <- testing(split) # won't be checking this till testing for proper results

#alright now recipe with upsampling and dummy variables
library(themis)

city_recipe <- recipe(critical ~ ., data = city_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_upsample(critical, over_ratio = 1) 

cls_metrics <- metric_set(yardstick::accuracy, yardstick::precision, yardstick::sens, yardstick::f_meas,
                          yardstick::spec)

#first the logistic regression and we use step AIC as there is some multi-collinearity

prep_train <- prep(city_recipe, training = city_train)
train_baked <- bake(prep_train, new_data = city_train)

library(MASS) 
# Full logistic model on baked data
full_glm <- glm(critical ~ ., data = train_baked, family = binomial())

# Stepwise AIC (both directions)
step_glm <- stepAIC(full_glm, direction = "both", trace = FALSE)

AIC(full_glm)
AIC(step_glm)

summary(step_glm)
vif(step_glm)
#there is still collinearity between X3 and X4, so step-wise model is not the best model, but for now let's just use this

reduced_glm_X3 <- glm(critical ~ . - X3, data = train_baked, family = binomial())
AIC(reduced_glm_X3)
reduced_glm_X4 <- glm(critical ~ . - X4, data = train_baked, family = binomial())
AIC(reduced_glm_X4)

#Since AIC is slightly lower for the stepwise model, it has marginally better fit-to-complexity balance than the full model.
#The improvement is very small, so the selection didn’t drastically change the explanatory power 
#it likely removed one or two variables that weren’t contributing much, mostly to reduce redundancy from the multicollinearity seen in Q3.
#well let's just use step-wise model
train_probs <- predict(step_glm, newdata = train_baked, type = "response")
thresholds <- seq(0.1, 0.9, by = 0.05)

threshold_metrics <- purrr::map_dfr(thresholds, function(t) {
  preds <- ifelse(train_probs > t, "yes", "no") %>% factor(levels = c("no","yes"))
  tibble(
    threshold = t,
    accuracy  = accuracy_vec(train_baked$critical, preds),
    sens      = sens_vec(train_baked$critical, preds),
    spec      = spec_vec(train_baked$critical, preds),
    precision = precision_vec(train_baked$critical, preds),
    f1        = f_meas_vec(train_baked$critical, preds)
  )
})

threshold_metrics

#I am just using best f-1 score here as we don't know the consequences of FP and FN
#we can prioritize sens or spec based on the consequence and for now I am choosing best F1 score as there is no info
best_t <- threshold_metrics %>% arrange(desc(f1)) %>% slice(1) %>% pull(threshold)
best_t

# Now for test set after baking with same recipe
test_baked <- bake(prep_train, new_data = city_test)
test_probs <- predict(step_glm, newdata = test_baked, type = "response")
test_preds <- ifelse(test_probs > best_t, "yes", "no") %>% factor(levels = c("no","yes"))

test_results_log <- tibble(
  accuracy  = accuracy_vec(test_baked$critical, test_preds),
  sens      = sens_vec(test_baked$critical, test_preds),
  spec      = spec_vec(test_baked$critical, test_preds),
  precision = precision_vec(test_baked$critical, test_preds),
  f1        = f_meas_vec(test_baked$critical, test_preds)
)

test_results_log

#now for next model, let's create a new split just in case
#tunable model would be randomn forest(chose this as it is an ensembling model and I like this more than XGboost)
#let's just tune all parametres except trees
#created new splits because I was not sure if using he same data would result in data leakage.

set.seed(321)

split_rf <- initial_split(city_critical, prop = 0.8, strata = critical)
city_train_rf <- training(split_rf)
city_test_rf  <- testing(split_rf)

city_recipe_rf <- recipe(critical ~ ., data = city_train_rf) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_upsample(critical, over_ratio = 1)

#larger values took a lot of time so I am choosing small values(Please understand, It took huge amount of my time)
rf_spec <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 500  
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

wf_city_rf <- workflow() %>%
  add_recipe(city_recipe_rf) %>%
  add_model(rf_spec)

city_folds_rf <- vfold_cv(city_train_rf, v = 5, strata = critical)

param_set_rf <- hardhat::extract_parameter_set_dials(wf_city_rf) %>%
  update(
    mtry  = mtry(range = c(2L, 6L)),
    min_n = min_n(range = c(5L, 30L))
  )

grid_rf <- grid_regular(param_set_rf, levels = 5)

metrics_set <- metric_set(yardstick::roc_auc, yardstick::accuracy, yardstick::precision, yardstick::sens, yardstick::f_meas, yardstick::spec)

rf_tuned <- tune_grid(
  wf_city_rf,
  resamples = city_folds_rf,
  grid = grid_rf,
  metrics = metrics_set
)

best_rf <- select_best(rf_tuned, metric = "roc_auc")
best_rf
#mtry=2 and min_n=30

# Finalizing workflow & fit on training set
final_rf <- finalize_workflow(wf_city_rf, best_rf) %>% fit(city_train_rf)

# Threshold tuning on training set for RF
train_probs_rf <- predict(final_rf, new_data = city_train_rf, type = "prob")$.pred_yes
thresholds <- seq(0.1, 0.9, by = 0.05)

threshold_metrics_rf <- purrr::map_dfr(thresholds, function(t) {
  preds <- ifelse(train_probs_rf > t, "yes", "no") %>% factor(levels = c("no","yes"))
  tibble(
    threshold = t,
    accuracy  = accuracy_vec(city_train_rf$critical, preds),
    sens      = sens_vec(city_train_rf$critical, preds),
    spec      = spec_vec(city_train_rf$critical, preds),
    precision = precision_vec(city_train_rf$critical, preds),
    f1        = f_meas_vec(city_train_rf$critical, preds)
  )
})

threshold_metrics_rf

# same reason as above I am choosing better F1 score here
best_t_rf <- threshold_metrics_rf %>% arrange(desc(f1)) %>% slice(1) %>% pull(threshold)
best_t_rf
# going to test data
rf_probs <- predict(final_rf, new_data = city_test_rf, type = "prob")$.pred_yes
rf_preds <- ifelse(rf_probs > best_t_rf, "yes", "no") %>% factor(levels = c("no","yes"))

rf_test_results <- tibble(
  accuracy  = accuracy_vec(city_test_rf$critical, rf_preds),
  sens      = sens_vec(city_test_rf$critical, rf_preds),
  spec      = spec_vec(city_test_rf$critical, rf_preds),
  precision = precision_vec(city_test_rf$critical, rf_preds),
  f1        = f_meas_vec(city_test_rf$critical, rf_preds)
)

rf_test_results

# let's compare
final_comparison <- tibble(
  Model = c("Stepwise Logistic Regression", "Random Forest"),
  accuracy = c(test_results_log$accuracy, rf_test_results$accuracy),
  sens = c(test_results_log$sens, rf_test_results$sens),
  spec = c(test_results_log$spec, rf_test_results$spec),
  precision = c(test_results_log$precision, rf_test_results$precision),
  f1 = c(test_results_log$f1, rf_test_results$f1)
)

final_comparison

#well the results could change based on the threshhold values choosen, as I dont have information
#i just went with best F1 score and not prioritizing sens or spec
#if I take criticality then FN might be important where we classify it as not critical but it is 
#In that thought process both models have good sens.but there will be a lot of false alarms in both cases(logistic will have way more).
#both have accuracy equal to an Naive model which predicts all as no, but that too is based on our requirement we can change the threshold.
#I am assuming this is a project just to test our concepts.


#maybe the Rf is overfitting so we have to decrease number of trees and keep the other values constant


rf_spec_small <- rand_forest(
  mtry  = best_rf$mtry,
  min_n = best_rf$min_n,
  trees = 200  # reduced from 500 to limit overfitting
) %>%
  set_engine("ranger") %>%
  set_mode("classification")


final_rf_small <- workflow() %>%
  add_recipe(city_recipe_rf) %>%
  add_model(rf_spec_small) %>%
  fit(city_train_rf)

# Threshold tuning on training set for smaller RF
train_probs_rf_small <- predict(final_rf_small, new_data = city_train_rf, type = "prob")$.pred_yes

threshold_metrics_rf_small <- purrr::map_dfr(thresholds, function(t) {
  preds <- ifelse(train_probs_rf_small > t, "yes", "no") %>% factor(levels = c("no","yes"))
  tibble(
    threshold = t,
    accuracy  = accuracy_vec(city_train_rf$critical, preds),
    sens      = sens_vec(city_train_rf$critical, preds),
    spec      = spec_vec(city_train_rf$critical, preds),
    precision = precision_vec(city_train_rf$critical, preds),
    f1        = f_meas_vec(city_train_rf$critical, preds)
  )
})
threshold_metrics_rf_small
best_t_rf_small <- threshold_metrics_rf_small %>% arrange(desc(f1)) %>% slice(1) %>% pull(threshold)
#best_t_rf_small<- 0.45
#0.45 gives different values which is not overfitting

# Evaluate on test set
rf_probs_small <- predict(final_rf_small, new_data = city_test_rf, type = "prob")$.pred_yes
rf_preds_small <- ifelse(rf_probs_small > best_t_rf_small, "yes", "no") %>% factor(levels = c("no","yes"))

rf_test_results_small <- tibble(
  accuracy  = accuracy_vec(city_test_rf$critical, rf_preds_small),
  sens      = sens_vec(city_test_rf$critical, rf_preds_small),
  spec      = spec_vec(city_test_rf$critical, rf_preds_small),
  precision = precision_vec(city_test_rf$critical, rf_preds_small),
  f1        = f_meas_vec(city_test_rf$critical, rf_preds_small)
)


final_comparison_2 <- tibble(
  Model = c("Stepwise Logistic Regression", "Random Forest (500 trees)", "Random Forest (200 trees)"),
  accuracy = c(test_results_log$accuracy, rf_test_results$accuracy, rf_test_results_small$accuracy),
  sens = c(test_results_log$sens, rf_test_results$sens, rf_test_results_small$sens),
  spec = c(test_results_log$spec, rf_test_results$spec, rf_test_results_small$spec),
  precision = c(test_results_log$precision, rf_test_results$precision, rf_test_results_small$precision),
  f1 = c(test_results_log$f1, rf_test_results$f1, rf_test_results_small$f1)
)

final_comparison_2

#well it is still overfitting(It is technically not) and the issue is not trees but the threshold we chose, 
#so we can choose some other threshold and it won't give overfitting values, 
#but bottom line based on the metrics we need we can choose optimal threshold and  get the metrics.
#my goal was to get values at different thresh and compare, as this is a hypothetical data 
#I am not going deep to see what metrics are more valuable and which thresh to use as I don't know what the severity is.
#so just leaving it at F1 score.(even though it is kinda overfitting)


#@# --END OF ANSWER_4--
#@#=================================
#@# --START OF QUESTION_5--
#@# Title: Feature Importance Analysis for Critical Condition Prediction
#@# Referring to your interpretable model(s), describe the extent to which 
#@# its input (predictors) significantly impact its outputs (predictions) of whether Y>112.
#@# Describe what you observe. Justify what you do.
#@# --END OF QUESTION_5--
#@# --START OF ANSWER_5--
#@# PASTE R CODE BELOW HERE

#the interpretable model used is step-wise model
summary(step_glm)

#OR > 1 means  increases odds of being in the yes class
#OR < 1 means decreases odds of being in the yes class
#OR = 1 means no effect


library(broom)

coef_df <- tidy(step_glm) %>%
  filter(term != "(Intercept)") %>% 
  mutate(
    odds_ratio = exp(estimate),
    effect_dir = ifelse(odds_ratio > 1, "Positive", "Negative"),
    abs_dist = abs(odds_ratio - 1)
  ) %>%
  arrange(desc(abs_dist))


ggplot(coef_df, aes(x =odds_ratio , y = reorder(term, abs_dist), fill = effect_dir)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  coord_flip() +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "red") +
  labs(
    title = "Feature Importance via Odds Ratios",
    x = "Odds Ratio",
    y = "predictor",
    fill = "Effect Direction"
  ) +
  theme_minimal(base_size = 14)

or_table <- broom::tidy(step_glm, conf.int = TRUE, conf.level = 0.95, exponentiate = TRUE) %>%
  dplyr::filter(term != "(Intercept)") %>%
  dplyr::transmute(
    Predictor = term,
    OR = estimate,
    CI_low = conf.low,
    CI_high = conf.high,
    p_value = p.value
  ) %>%
  dplyr::arrange(desc(abs(OR - 1)))
or_table
#this gives the 95% CI for the predictors, and we can make a proper judgement based on this.

#from this graph we can clearly see cases where OR>1 which will have positive effect.
#for X5 odds ratio is 1.46 which means each one-unit increase is associated with a 46% increase in odds of the “Yes” outcome..
#for X4 and X3 odds ratio is 1.20 and 1.12 which means Moderate/small positive effect on critical outcome.
#X6 has odds-ratio of 0.73 which means higher values reduce odds by ~27%.
#X1_no and X2_hot(dummy variables) have odds-ratio of 0.24 and 0.016 which strongly/very strongly decreases odds

#X3 and X4 have high VIF values (>8), suggesting multicollinearity. This may inflate their standard errors, 
#making the effect size estimates less stable
#Odds ratios describe multiplicative changes in odds, not probabilities. 
#The practical impact on probability depends on the baseline odds
# for example, if baseline p=0.20 (odds 0.25), a one-unit increase in X5 yields odds 0.25×1.46=0.365 , that is p≈0.27.







# I tried various things for each question in this lab and did my best.(did not go deep in few as my laptop was not able to habdle them)
#did not take AI assistance much, just took it in few places as my code was not working in those places.
#Thank you for everything Professor and Garrik. It was a great class and assignments were really educational.


#@# --END OF ANSWER_5--
#@#=================================