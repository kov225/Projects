#@# Regression and Classification Lab 4 Submission Template
#@# Instructions:
#@# 1. Do not modify any line beginning with #@#
#@# 2. Do not begin any line in your solution programs or comments with #@#
#@# 3. Paste your R code below the line PASTE R CODE BELOW HERE for each answer.
#@# 4. Ensure each answer corresponds to the correct question number.
#@# 5. Provide code for all 5 questions, even if incomplete, to avoid parsing errors.
#@# 6. Save this file as a plain .txt file and upload via the Google Form.
#@#=================================
#@# --START OF QUESTION_1--
#@# Title: World Happiness Linear Regression Analysis (Statistics Framework)
#@# You are a policy analyst for the United Nations investigating factors that contribute to national happiness levels across countries.
#@#
#@# A) Build a simple linear regression model predicting happiness scores (Ladder) based on economic prosperity (LogGDP). 
#@#    Interpret the slope coefficient in policy terms and assess the model's explanatory power using R-squared.
#@#
#@# B) Expand your model to include SocialSupport and LifeExpectancy as additional predictors. 
#@#    Compare this multiple regression model to the simple model using appropriate model comparison techniques (AIC/BIC). 
#@#    Interpret the coefficients, their confidence intervals, and their p-values for each predictor.
#@#
#@# C) Conduct comprehensive regression diagnostics for your multiple regression model. 
#@#    Check for influential observations using leverage and Cook's distance.
#@#    Assess requirements for regression both visually and/or formally:
#@#    e.g. normally distributed residuals, homoscedasticity, absence of multicollinearity.
#@#
#@# D) Use stepwise model selection to determine the optimal set of predictors from all available variables 
#@#    (LogGDP, SocialSupport, LifeExpectancy, Freedom, Generosity, Corruption). 
#@#    Compare the final selected model with your model from Part B and justify which model you would recommend for policy analysis.
#@# --END OF QUESTION_1--
#@# --START OF ANSWER_1--
#@# PASTE R CODE BELOW HERE

library(tidyverse)
library(broom)
library(car)
library(MASS)      
library(ggplot2)
library(readr)
library(caret)
library(purrr)
library(tibble)

#first the datframe
happiness<- read.csv("Worldhappinessreport2024clean.csv")
head(happiness)
dim(happiness)
#checking for missing values
sum(is.na(happiness))        
colSums(is.na(happiness))

#as there are no missiing values the data is clean and dosn't need any other pre-processing

#1-A
model_1_A <- lm(Ladder ~ LogGDP, data = happiness)
summary(model_1_A)

#The co-ffecient we got is 2.22 and that means 
#For every 1 unit increase in LogGDP, a country's Happiness Score (output variable) is expected to increase by approximately 2.22 units.

#p-value is extremely small that means 
#it is statistically significant
#what I think statistically significant means is that association between these 2 variables is unlikely to be by chance.(not a fluke)

#Coming to model's explanatory power with R-squared value, which is around 0.63, and that means 
#63.2% of the variance in Happiness Scores across countries is explained solely by LogGDP.
#isn't that pretty high for a single variable model?
#maybe LogGDP is strongly associated to Happiness Score


#1-B

model_1_B <- lm(Ladder ~ LogGDP + SocialSupport + LifeExpectancy, data = happiness)
summary(model_1_B)

confint(model_1_A) 
#We are 95% confident that the true effect of LogGDP lies between 1.93 and 2.51.
#We are 95% confident that the Intercept lies between 2.05 and 2.89
confint(model_1_B) 

#The coefficient for LogGDP is now 0.89 that means,
#For every 1 unit increase in LogGDP, happiness is  increased by 0.89 units,
#holding SocialSupport and LifeExpectancy constant.
#from p value we can say it is still statistically significant (p = 0.0016).
#CI: [0.34, 1.44]- We are 95% confident that the true effect of LogGDP lies between 0.34 and 1.44 for this model.

# it has decreased from 2.22 to 0.89 with the addition of other variables
#maybe the impact overlaps with other variables

# Now for  SocialSupport: the coefficient is 1.55.
#It means that when SocialSupport increases by 1 unit, happiness jumps by 1.55 units.
#It's also highly significant (p < 0.00001).
#CI: [1.05, 2.05]-We are 95% confident that the true effect of SocialSupport lies between 1.05 and 2.05

#LifeExpectancy has a coefficient of 1.14, but the p-value is 0.08 and CI includes 0.
#So it's not statistically significant , still possibly meaningful.
#CI: [-0.14, 2.41]- We are 95% confident that the true effect of LifeExpectancy lies between -0.14 and 2.41 and it includes 0 in this interval.

#but we don’t have enough statistical confidence here to treat it as a
#good independent variable once GDP and social support are accounted for

#after analyzing we can say social support has the highest impact on Happiness score, and 
#follwed by LogGDp and then LifeExpectancy. 
#result makes sense if we think about it logically.

AIC(model_1_A, model_1_B) #289.15, 253.14

BIC(model_1_A, model_1_B) #297.88, 267.7

#The AIC and BIC values for are both lower for multiple variable model which means 
#the multiple variable model(model_1_B) is better than model_1_A
#so adding other variables was worth it.

#1-C
#let's plot graphs first for ananlysis 
par(mfrow = c(2, 2))
plot(model_1_B)

#Residuals are reasonably normal in my opinion
#there seems to be few influencial points
library(car)
influencePlot(model_1_B, id.method="identify")
#this shows that there is some leverage and influencial points wich are 123,124,130,132,136.

#for collinearity we use vif
vif(model_1_B)
#using 5 as threshhold we can safely say that there is no multi-collinearity

#Leverage
hatvalues(model_1_B)

#Cook's distance
cooks.distance(model_1_B)



#influence measures 
influence_measures <- influence.measures(model_1_B)
summary(influence_measures)
#this gives even better understanding for influencial points
#The most concerning observations are 128, 130, 132, 136.
#Obs 124 & 111 also high leverage but with smaller residual effects.
#Several others (74, 81, 84, 87, 122, 123, 127) show mild influence.

#Shapiro–Wilk test on the model residuals to assess normality
shapiro.test(residuals(model_1_B)) 
#as p-value is less than 0.05 we can say it is not normal

#homoscedasticity

library(lmtest)
bptest(model_1_B) #Breusch–Pagan test-Null hypothesis: Constant variance of residuals (homoscedasticity)
#as p-value is less than 0.05 we can reject null hypothesis which is no homoscedasticity.



# 4. for visualizing
plot(hatvalues(model_1_B), type="h", main="Leverage")
abline(h = 2*mean(hatvalues(model_1_B)), col="red", lty=2)

plot(cooks.distance(model_1_B), type="h", main="Cook's Distance")
abline(h = 4/(nrow(df)-length(model_1_B$coefficients)), col="red", lty=2)

#1-D

#first we create a model with all variables
model_full <- lm(Ladder ~ LogGDP + SocialSupport + LifeExpectancy + Freedom + Generosity + Corruption, data = happiness)
summary(model_full)


model_1_D <- stepAIC(model_full, direction = "both", trace = FALSE)
summary(model_1_D)
#the new model took all predictors except Genororsity

AIC(model_1_B, model_1_D)
BIC(model_1_B, model_1_D)
#I would recommend the AIC model(mode_1D) as it has lower AIC and BIC values
 
#@# --END OF ANSWER_1--
#@#=================================
#@# --START OF QUESTION_2--
#@# Title: World Happiness Linear Regression Analysis (Tidymodels Framework)
#@# You are now implementing the same happiness analysis using modern machine learning workflows to ensure reproducible and robust results.
#@#
#@# A) Using the tidymodels framework, create a workflow that includes a linear regression model specification and 
#@#    a recipe for predicting happiness (Ladder) using LogGDP, SocialSupport, and LifeExpectancy. 
#@#    Fit this workflow to the data and extract the model coefficients.
#@#
#@# B) Implement 5-fold cross-validation to assess model performance. Calculate and interpret the cross-validated RMSE, MAE, and R-squared values. 
#@#    Compare these results with the simple fitted model performance.
#@#
#@# C) Create predictions for new hypothetical countries with LogGDP=10.5, SocialSupport=0.8, and LifeExpectancy=70. 
#@#    Calculate both point predictions and 95% confidence intervals. Interpret what these intervals tell you about prediction uncertainty.
#@#
#@# D) Split the data into training (80%) and testing (20%) sets. Train your model on the training data and evaluate performance on the test set.
#@#     Compare the test set performance with your cross-validation results in (B) and explain any differences.
#@# --END OF QUESTION_2--
#@# --START OF ANSWER_2--
#@# PASTE R CODE BELOW HERE

#2-A
#I am a little confused, question said to use it on data and dint specify training data and didnt ask to split data in the start
#but if we fit on full data there would be data leakage, so I am going to split right from the beginning


library(tidymodels) 
#not normalizing data, Normalizing is not changing much as the data is already scaled , so I dont want to lose easy interpretability
happiness_recipe <- recipe(Ladder ~ LogGDP + SocialSupport + LifeExpectancy, data = happiness) 
set.seed(42)
data_split <- initial_split(happiness, prop = 0.8)
happiness_train <- training(data_split)
happiness_test  <- testing(data_split)

model_2_A <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")


lm_workflow <- workflow() %>%
  add_model(model_2_A) %>%
  add_recipe(happiness_recipe)

lm_fit <- fit(lm_workflow, data = happiness_train)

tidy(lm_fit)
train_results <- predict(lm_fit, train_data) %>%
  bind_cols(dplyr::select(train_data, Ladder))
train_results
train_metrics <- metrics(train_results, truth = Ladder, estimate = .pred)
train_metrics



#2-B

# Setting seed
set.seed(18)
folds <- vfold_cv(happiness, v = 5)

# Fit with resampling
happiness_results <- fit_resamples(
  lm_workflow,
  resamples = folds,
  metrics = metric_set(rmse, mae, rsq)
)

# Collect metrics
collect_metrics(happiness_results)

#comparing these metrics with part A we get 
#The cross‑validated RMSE is slightly lower than the training RMSE
#cross‑validated MAE is slightly lower than the training MAE
#R-squared value also slightly dropped from training to cross-validation
#But overall they are identical as the drops are very low in value

#2-C

# taking finland as base country
base_country <- happiness_train %>%
  dplyr::filter(Country.name == "Finland") %>%
  dplyr::select(LogGDP, SocialSupport, LifeExpectancy)#I have to re-install dplyr i guess
head(base_country)
# ex-1 -10% GDP, +10% Life Expectancy
new_country_1 <- base_country %>%
  mutate(
    LogGDP = LogGDP * 0.9,
    LifeExpectancy = LifeExpectancy * 1.1
  )
head(new_country_1)
# Scenario 2: +10% GDP, -10% Life Expectancy
new_country_2 <- base_country %>%
  mutate(
    LogGDP = LogGDP * 1.1,
    LifeExpectancy = LifeExpectancy * 0.9
  )

# Predictions (point + CI)
pred_1 <- predict(lm_fit, new_country_1, type = "conf_int") %>%
  bind_cols(predict(lm_fit, new_country_1))

pred_2 <- predict(lm_fit, new_country_2, type = "conf_int") %>%
  bind_cols(predict(lm_fit, new_country_2))

pred_1 #	[6.52, 7.09]
pred_2 # [6.59, 7.17]

max(happiness$Ladder)
#interpreting results for pred-1
#The model predicts a happiness ladder score of 6.81 under the given conditions, that's prety high as the max score was 7.7
#the 95% confidence interval is [6.52, 7.09], that means if repeated sample are studied with this values then the values mostly lie in this interval
#While exact values may fluctuate, the result  indicates a stable outcome in the high‑6 range on the  happiness ladder.

#now the results make sense, wasted a lot of time on that question example.it gave me min 7.4 and max 178 for the example in question.

#2-D

# Fit on training data
final_fit <- fit(lm_workflow, data = happiness_train)

# Predict on test data
test_results <- predict(final_fit, happiness_test) %>%
  bind_cols(dplyr::select(happiness_test, Ladder))

# Evaluate performance on test data
metrics(test_results, truth = Ladder, estimate = .pred)
#Comparing with part B
#as the rmse is lower, the model did a  better job on unseen data 
#MAE is lower too which indicates predictions are closer on average in the test dataset as compared to part B 
#R squared value droped from 71.2 to 69.3, but the model still generalizes well.

max(happiness$Ladder)#just for understanding

#@# --END OF ANSWER_2--
#@#=================================
#@# --START OF QUESTION_3--
#@# Title: Obesity Classification with Logistic Regression (Statistics Framework)
#@# You are a public health researcher developing a screening tool to identify individuals at risk of being overweight or obese based on lifestyle factors.
#@# When building models: Convert categorical variables to factors. Keep only complete cases for analysis variables.
#@#    
#@# A) Create a binary outcome variable classifying individuals as "Normal Weight" vs "At Risk" (combining all overweight and obesity categories). 
#@#    Build a simple logistic regression model predicting this outcome using physical activity frequency (FAF). 
#@#    Interpret the coefficient in terms of odds ratios.
#@#
#@# B) Expand your model to additionally include Age, frequent consumption of high caloric food (FAVC), and frequency of vegetable consumption (FCVC) as predictors. 
#@#    Calculate and interpret the odds ratios for each predictor, including 95% confidence intervals.
#@#
#@# C) Evaluate your model's classification performance by creating a confusion matrix using a 0.5 probability threshold. 
#@#    Calculate accuracy, sensitivity, specificity, and precision. Interpret these metrics in the context of public health screening.
#@#
#@# D) Investigate how changing the classification threshold affects model performance. 
#@#    Test thresholds of 0.3, 0.4, 0.5, 0.6, and 0.7, and determine which threshold would be most appropriate for a public health screening program. 
#@#    Justify your choice considering the costs of false positives vs false negatives.
#@# --END OF QUESTION_3--
#@# --START OF ANSWER_3--
#@# PASTE R CODE BELOW HERE

#loading dataset
obese <- read.csv("ObesityDataSet_raw_and_data_sinthetic.csv")
#checking for na values
head(obese)
dim(obese)
#checking for missing values
sum(is.na(obese))        
colSums(is.na(obese))
str(obese)
#as there are no na values we don't need to clean data

# data pre-processing
# using factors and converting string to numeric(realized this too late)
obese <- obese %>%
  mutate(FAVC = as.factor(FAVC),
         FCVC = as.numeric(FCVC),
         NObeyesdad = as.factor(NObeyesdad))
obese %>% count(NObeyesdad)

#taking positive class as AtRisk
obese <- obese %>%
  mutate(AtRisk = ifelse(NObeyesdad %in% c("Overweight_Level_I", "Overweight_Level_II", 
                                           "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"), 1, 0)) %>%
  mutate(AtRisk = factor(AtRisk, levels = c(0,1), labels = c("Normal", "AtRisk")))

#3-A
#using logistic regression
model_3_A <- glm(AtRisk ~ FAF, data = obese, family = binomial)
summary(model_3_A)

# Odds Ratio
exp(coef(model_3_A))

#gives 95% confidence intervals
exp(cbind(OR = coef(model_3_A), confint(model_3_A)))

#When FAF = 0, the odds of being At Risk are about 4.5 times higher than the odds of being Normal Weight.
#Each 1‑unit increase in FAF multiplies the odds of being At Risk by 0.64
#95% CI is [0.57,0.72]  this effect is statistically significant.

#3-B

model_3_B <- glm(AtRisk ~ FAF + Age + FAVC + FCVC, data = df_clean, family = binomial)
summary(model_3_B)

# Odds Ratios and 95% CI
exp(coef(model_3_B))
exp(cbind(OR = coef(model_3_B), confint(model_3_B)))

#	Each unit increase in FAF reduces the odds of being At Risk by 23% (1 − 0.77). 
#Each year increase in age increases the odds of being At Risk by 23%
#People who frequently consume high-calorie foods have 3.5× the odds of being At Risk compared to those who don’t
#Frequency of vegetable consumption shows no significant effect as CI crosses 1.

#3-C
# first finding predicted probabilities
pred_probs <- predict(model_3_B, type = "response")

# Class predictions at 0.5 threshold
pred_class <- ifelse(pred_probs >= 0.5, "AtRisk", "Normal") %>% as.factor()

# Confusion Matrix
cm_3_C <- caret::confusionMatrix(pred_class, obese$AtRisk, positive = "AtRisk")

# Extract metrics
TP <- cm_3_C$table[2,2]
TN <- cm_3_C$table[1,1]
FP <- cm_3_C$table[1,2]
FN <- cm_3_C$table[2,1]

accuracy <- (TP + TN) / (TP + TN + FP + FN)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
precision <- TP / (TP + FP)

metrics_3_C <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision"),
  Value = round(c(accuracy, sensitivity, specificity, precision), 3)
)
metrics_3_C
#The model classifies about 77% of individuals correctly.
#Of all true AtRisk individuals, the model correctly identifies about 79%
#Of all true Normal individuals, about 66% are correctly identified as Normal
#Of those classified as AtRisk, about 95% are truly AtRisk

#from these metrics we can say that our moel is pretty reliable
#3-D
#first varying thresholds
thresholds_3_D <- c(0.3, 0.4, 0.5, 0.6, 0.7)

# Evaluating perfromance for these thresholds
threshold_results_3_D <- map_df(thresholds_3_D, function(thresh) {
  pred <- ifelse(pred_probs >= thresh, "AtRisk", "Normal") %>% 
    factor(levels = c("Normal", "AtRisk"))
  
  cm_3_D <- caret::confusionMatrix(pred, obese$AtRisk, positive = "AtRisk")
  
  tibble(
    Threshold   = thresh,
    Accuracy    = as.numeric(cm_3_D$overall["Accuracy"]),
    Sensitivity = as.numeric(cm_3_D$byClass["Sensitivity"]),
    Specificity = as.numeric(cm_3_D$byClass["Specificity"]),
    Precision   = as.numeric(cm_3_D$byClass["Precision"])
  )
})

threshold_results_3_D

#visualizing
ggplot(threshold_results, aes(x = Threshold)) +
  geom_line(aes(y = Sensitivity, color = "Sensitivity"), size = 1.2) +
  geom_line(aes(y = Specificity, color = "Specificity"), size = 1.2) +
  labs(title = "Threshold vs Sensitivity & Specificity",
       y = "Metric Value") +
  scale_color_manual(values = c("Sensitivity" = "blue", "Specificity" = "red")) +
  theme_minimal()
#this graph shows sens and spec with increasing threshold, good for visualizing
#for screening we need high sensitivity because False positives means extra tests, resource strain, anxiety.
# and False negatives means  people go untreated, higher morbidity/mortality.so false negatives are more important.
#we can choose  0.3 but we can choose 0.4 too as sens is still 96% with higher metrics.(all other metrics)
#personally I would choose 0.4




#@# --END OF ANSWER_3--
#@#=================================
#@# --START OF QUESTION_4--
#@# Title: Obesity Classification with Logistic Regression (Tidymodels Framework)
#@# You are implementing the obesity classification system using modern ML workflows to ensure robust model evaluation and deployment.
#@#
#@# A) Using tidymodels, create a workflow with a logistic regression model and recipe for the binary obesity classification problem. 
#@#    Include physical activity frequency (FAF), Age, frequent consumption of high caloric food (FAVC), and frequency of vegetable consumption (FCVC) as predictors. 
#@#    Ensure categorical variables are properly encoded in your recipe.
#@#
#@# B) Implement 5-fold cross-validation with stratified sampling to evaluate model performance. 
#@#    Calculate cross-validated accuracy, ROC AUC, sensitivity, and specificity. 
#@#    Explain why stratified sampling is important for this classification problem.
#@#
#@# C) Create and interpret ROC curves for your model. Calculate the area under the ROC curve (AUC) and explain what this metric tells you about model performance. 
#@#    Identify the optimal threshold using the ROC curve.
#@#
#@# D) Compare your tidymodels results with the traditional statistics approach from Question 3. 
#@#    Create final predictions on a test set and generate a comprehensive confusion matrix with all classification metrics on the test set. 
#@#    Discuss any differences between the two approaches.
#@# --END OF QUESTION_4--
#@# --START OF ANSWER_4--
#@# PASTE R CODE BELOW HERE

#as we are using the same data from above and it has already been preprocessed, we dont need to pre process it here agaain
set.seed(111)
obese_split <- initial_split(obese, prop = 0.8, strata = AtRisk)
obese_train <- training(obese_split)
obese_test <- testing(obese_split)

#4-A
#going in order

# initial split
set.seed(110)
obese_split <- initial_split(df, prop = 0.8, strata = AtRisk)
obese_train <- training(obese_split)
obese_test  <- testing(obese_split)
#  recipe with encoding
obese_recipe <- recipe(AtRisk ~ FAF + Age + FAVC + FCVC, data = obese_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())


log_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# workflow
obese_wf <- workflow() %>%
  add_model(logistic_spec) %>%
  add_recipe(obese_recipe)

#4-B

set.seed(113)

# fit with resampling
obese_folds <- vfold_cv(obese_train, v = 5, strata = AtRisk)

obese_results <- fit_resamples(
  obese_wf,
  resamples = obese_folds,
  metrics = yardstick::metric_set(
    yardstick::accuracy,
    yardstick::roc_auc,
    yardstick::sensitivity,
    yardstick::specificity
  ),
  control = control_resamples(save_pred = TRUE)
)


collect_metrics(obese_results)

#Without stratification, folds may accidentally have very different proportions of Normal vs AtRisk
#we might be a fold with only the majority class which is not good.
#that might increase accuracy artifically and in a naive manner.

#4-C

model_4_C <- fit(obese_wf, data = obese_train)


train_preds <- predict(model_4_C, obese_train, type = "prob") %>%
  bind_cols(obese_train %>% dplyr::select(AtRisk))


roc_obj <- roc(response = train_preds$AtRisk,
               predictor = train_preds$.pred_AtRisk)

# Plot ROC curve
plot(roc_obj, main = "ROC Curve - Training Set")
#A perfect model would hug the top-left corne and the diagnol line in AUC=0.5, which is randomn guessing 
auc_val <- auc(roc_obj)
auc_val
#this value of 0.794 says that our model is   better than random,
#with good separation between Normal and AtRisk.

# Find optimal threshold using Youden's J (stack overflow and gpt)
best_thresh <- coords(roc_obj, x = "best", best.method = "youden",
                      ret = c("threshold", "sensitivity", "specificity"))
best_thresh
#this is the optimal as this maximizes the balance between Sensitivity and Specificity.

#4-D

# Predictions on test set
test_preds <- predict(model_4_C, obese_test, type = "prob") %>%
  bind_cols(predict(model_4_C, obese_test)) %>%
  bind_cols(obese_test %>% dplyr::select(AtRisk))

test_probs <- predict(model_4_C, obese_test, type = "prob") %>%
  bind_cols(predict(model_4_C, obese_test)) %>%
  bind_cols(obese_test %>% dplyr::select(AtRisk))

# Apply optimal threshold
opt_threshold <- as.numeric(best_thresh$threshold)

test_preds <- test_preds %>%
  mutate(pred_class_opt = ifelse(.pred_AtRisk >= opt_threshold, "AtRisk", "Normal"),
         pred_class_opt = factor(pred_class_opt, levels = levels(AtRisk)))

# Confusion matrix (optimal threshold)
conf_mat(test_preds, truth = AtRisk, estimate = pred_class_opt)

auc_test <- yardstick::roc_auc(test_preds, truth = AtRisk, .pred_AtRisk, event_level = "second")

auc_test
# Classification metrics
class_metrics <- yardstick::metric_set(
  yardstick::accuracy,
  yardstick::sensitivity,
  yardstick::specificity,
  yardstick::precision,
  yardstick::recall,
  yardstick::f_meas
)

final_metrics <- class_metrics(test_preds, truth = AtRisk, estimate = pred_class_opt)

# Combine AUC + other metrics
all_metrics <- bind_rows(auc_test, final_metrics)
print(all_metrics)


#in tidymodel approach we have  high Sensitivity, But Precision is low, meaning many false positives. But fewer AtRisk people are missed. 
#in 3rd question model it has better overall balance with higher Accuracy and Precision, so this is best for resource efficiency and has fewer false alarms.


#@# --END OF ANSWER_4--
#@#=================================
#@# --START OF QUESTION_5--
#@# Title: Addressing Class Imbalance in Obesity Classification
#@# You have discovered that your obesity dataset has class imbalance issues that may affect model performance and need to address this challenge.
#@#
#@# A) Analyze the distribution of your binary outcome variable (Normal Weight vs At Risk). 
#@#    Calculate the exact proportions and assess whether class imbalance is a concern. 
#@#    Explain how class imbalance could affect model performance and interpretation.
#@#
#@# B) Create a "naive" baseline model that always predicts the majority class. 
#@#    Calculate its accuracy and compare it with your logistic regression model from Question 4. 
#@#    Explain why accuracy alone may be misleading for imbalanced datasets.
#@#
#@# C) Implement different classification thresholds (0.2, 0.3, 0.4, 0.5, 0.6) and evaluate how they affect sensitivity and specificity. 
#@#    Create a plot showing the trade-off between these metrics. Recommend an optimal threshold for public health screening purposes.
#@#
#@# D) Calculate precision and recall for your recommended threshold from Part C. 
#@#    Compute the F1-score and explain why this metric might be more appropriate than accuracy for imbalanced classification problems. 
#@#    Discuss the practical implications of your final model for public health policy.
#@# --END OF QUESTION_5--
#@# --START OF ANSWER_5--
#@# PASTE R CODE BELOW HERE



#we are using the same data as above
#5-A

class_distribution <- obese %>%
  count(AtRisk) %>%
  mutate(Proportion = round(n / sum(n), 3))

class_distribution
#About 3 out of 4 individuals are classified as At Risk.this means the dataset is imbalanced toward the At Risk class
#this matters because A naive model could predict everyone as “At Risk” and still achieve ~74% accuracy
#if that happens then Specificity may drop which means that too many healthy individuals incorrectly flagged.which is waste of resouces
#if data is imbalanced, logistic regression will lean toward predicting At Risk, 
#inflating accuracy but underestimating true Normal cases.


#5-B
#naive model 
majority_class <- class_distribution %>%
  filter(n == max(n)) %>%
  pull(AtRisk)

naive_preds <- rep(majority_class, nrow(obese))
naive_accuracy <- mean(naive_preds == obese$AtRisk)

naive_accuracy
#even though accuracy is lower on test dataset for model in ques 3 it doesnt ignore the minor class and has higher sen and spec than naive model.
#which is more important than accuracy in this case.
#accuracy of 73% may look good but it is inflated accuracy by ignoring minor class.
#While the naive model achieves decent accuracy just by exploiting class imbalance, it is useless for public health screening 
#since it can’t identify normal individuals at all

#5-C
thresholds <- c(0.2, 0.3, 0.4, 0.5, 0.6)

threshold_metrics <- map_dfr(thresholds, function(thresh) {
  pred_class <- ifelse(test_probs$.pred_AtRisk >= thresh, "AtRisk", "Normal") %>%
    factor(levels = c("Normal", "AtRisk"))
  
  tibble(
    Threshold   = thresh,
    Accuracy    = accuracy_vec(test_probs$AtRisk, pred_class),
    Sensitivity = sensitivity_vec(test_probs$AtRisk, pred_class, event_level = "second"),
    Specificity = specificity_vec(test_probs$AtRisk, pred_class, event_level = "second"),
    Precision   = precision_vec(test_probs$AtRisk, pred_class, event_level = "second"),
    Recall      = recall_vec(test_probs$AtRisk, pred_class, event_level = "second"),
    F1_Score    = f_meas_vec(test_probs$AtRisk, pred_class, event_level = "second")
  )
})

print(threshold_metrics)

# Plot for trade off
results <- threshold_metrics %>%
  dplyr::select(Threshold, Sensitivity, Specificity) %>%
  tidyr::pivot_longer(cols = c(Sensitivity, Specificity), 
                      names_to = "Metric", values_to = "Value")

ggplot(results, aes(x = Threshold, y = Value, color = Metric)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Threshold vs Sensitivity & Specificity",
       y = "Metric Value") +
  scale_color_manual(values = c("Sensitivity" = "blue", "Specificity" = "red")) +
  theme_minimal()
#sens decreases with increasing thresh and spec increases.
# Recommended thresh
recommended_threshold <- 0.4

# i think it should be between 0.3 and 0.4 but from this case its better to take 0.4 as it has good sens and better metrics compared to 0.3

#5-D
#Precision, Recall, F1 at Recommended Threshold
pred_class_final <- ifelse(test_probs$.pred_AtRisk >= recommended_threshold, "AtRisk", "Normal") %>%
  factor(levels = c("Normal", "AtRisk"))

#could't get values for some reason with my code so took help 
precision_val <- precision_vec(test_probs$AtRisk, pred_class_final, event_level = "second")
recall_val    <- recall_vec(test_probs$AtRisk, pred_class_final, event_level = "second")
f1_val        <- f_meas_vec(test_probs$AtRisk, pred_class_final, event_level = "second")
acc_val       <- accuracy_vec(test_probs$AtRisk, pred_class_final)

final_results <- tibble(
  Threshold = recommended_threshold,
  Accuracy  = round(acc_val, 3),
  Precision = round(precision_val, 3),
  Recall    = round(recall_val, 3),
  F1_Score  = round(f1_val, 3)
)

final_results

#F1‑Score penalizes both false positives and false negatives, giving a balanced value.
#so it is a better metric for imbalanced dataset.
#and High Recall (95%) ensures very few AtRisk individuals go undetected
#With high Recall and decent Precision, the F1‑Score (0.845) confirms 
#the model is effective in detecting AtRisk individuals while controlling false alarms.

#in my opinion this model is Suitable for early detection campaigns,
#where missing AtRisk individuals is riskier than testing some healthy ones.(or any other case this condition is true)

#I think 0.4 is optimal, but opinions may vary.

#@# --END OF ANSWER_5--
#@#=================================

