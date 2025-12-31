#@# Regression and Classification Lab 5 Submission Template
#@# Instructions:
#@# 1. Do not modify any line beginning with #@#
#@# 2. Do not begin any line in your solution programs or comments with #@#
#@# 3. Paste your R code below the line PASTE R CODE BELOW HERE for each answer.
#@# 4. Ensure each answer corresponds to the correct question number.
#@# 5. Provide code for all 5 questions, even if incomplete, to avoid parsing errors.
#@# 6. Save this file as a plain .txt file and upload via the Google Form.
#@#=================================
#@# --START OF QUESTION_1--
#@# Title: Water Potability Data Preprocessing with Tidyverse and Recipes
#@# You are a data scientist for a water quality monitoring agency developing a classification system to predict water potability based on chemical and physical properties.
#@#
#@# A) Load the water_potability.csv dataset using tidyverse functions and conduct an initial exploratory analysis. 
#@#    Use dplyr and ggplot2 to examine the structure, summary statistics, and identify any missing values across all variables.
#@#    Calculate the proportion of potable vs non-potable water samples and assess if class imbalance is present.
#@#
#@# B) Create a tidymodels recipe to handle missing values appropriately for machine learning modeling. 
#@#    Use recipe steps like step_impute_mean() or step_impute_median() as appropriate.
#@#    Examine the distribution of each predictor variable using tidyverse visualization and identify any variables with significant skewness.
#@#
#@# C) Expand your recipe to apply appropriate transformations to address skewness in the predictor variables. 
#@#    Use recipe steps like step_log(), step_sqrt(), or other transformations as needed.
#@#    Also consider normalization and scaling steps such as step_normalize(), step_scale(), or step_center() to standardize your variables.
#@#    Create before-and-after visualizations using ggplot2 for the transformed variables by applying your recipe.
#@#
#@# D) Add steps to your recipe to address class imbalance if present using appropriate recipe functions.
#@#    Compare the original class distribution with your chosen balancing method.
#@#    Finalize your preprocessing recipe that will be used for all subsequent modeling questions and provide a summary of all recipe steps applied.
#@# --END OF QUESTION_1--
#@# --START OF ANSWER_1--
#@# PASTE R CODE BELOW HERE


library(tidyverse)
library(tidymodels)
library(e1071)   
library(dplyr)
getwd()
water <- read_csv("water_potability.csv")

head(water)
dim(water)
#checking for missing values
sum(is.na(water))        
colSums(is.na(water))

#There are a total of 1494 missing values out of which 
#491 are from ph
#781 from sulfate
#162 from Trihalomethanes


#1-A

# Structure & summary
glimpse(water)
sapply(water, class)
summary(water)
#there are few rows with ph values 0 which is impossible.Rest seems fine to my knowledge

# checking class distribution
class_prop <- water %>%
  count(Potability) %>%
  mutate(Proportion = n / sum(n))
class_prop
#There is class imbalance but it is not extreme


# visualization 
class_prop %>%
  ggplot(aes(x = factor(Potability), y = Proportion, fill = factor(Potability))) +
  geom_col() +
  scale_fill_manual(values = c("skyblue", "orange"),
                    labels = c("Not Potable", "Potable")) +
  labs(title = "Water Potability Distribution",
       x = "Potability", y = "Proportion") +
  theme_minimal()

water %>%
  pivot_longer(-Potability, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Predictor Variables")


#lets check skewness for all of them except portability(don't need for it)
water %>%
  select(-Potability) %>%
  summarise(across(everything(), ~ skewness(., na.rm = TRUE)))

#Solids column might need a log transformation as it is moderately-right skewed
#conductivity might be slightly right skewed(graph shows it too)

#1-B
#recipe
#as there are some clomuns which are sightly skweded I am going with step_impute_median()
water_recipe_1_B <- recipe(Potability ~ ., data = water) %>%
  step_impute_median(all_numeric_predictors()) #or can use (all_numeric(),-all_outcome()) as outcome is also numeric

# prep
water_prep_1_B <- prep(water_recipe_1_B, training = water)

#bake
data_1_B <- bake(water_prep_1_B, new_data = NULL)
head(data_1_B)

# Visualizing
data_1_B %>%
  pivot_longer(-Potability, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "lightgreen", color = "black") +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distributions After Median Imputation")

#There are few spikes in ph,sulfate and Trihalomethanes as the missing values are replaced with median.
#Rest of the graphs look same and should be same to as nothing changed in them.

# Skewness check again
skewness_1_B <- data_1_B %>%
  select(-Potability) %>%
  summarise(across(everything(), ~ skewness(., na.rm = TRUE))) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Skewness")

skewness_1_B

#slight changes(very less) in columns with missing values and rest are same.
#Missing values are succesfully filled with median values.

#1-C

# Updating recipe
water_recipe_1_C <- water_recipe_1_B %>%
  step_log(Solids, base = 10, offset = 1) %>%  
  step_normalize(all_numeric_predictors())

#only using it on solids as it is the only one with skewness greater than 0.5.
#used ofset to prevent log(0)

# updated prep
water_prep_1_C <- prep(water_recipe_1_C, training = water)

#baking
data_1_C <- bake(water_prep_1_C, new_data = NULL)
head(data_1_C)
#Yup, solids are log transformed. Last time made a mistake by not checking.

#skewness after log
skewness <- data_1_C %>%
  select(-Potability) %>%
  summarise(across(everything(), ~ e1071::skewness(., na.rm = TRUE))) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Skewness")

skewness
#now skewness is  -1.13 after using log, need to use some other transformation .

#using sqrt
water_recipe_sqrt <- water_recipe_1_B %>%
  step_sqrt(Solids) %>%
  step_normalize(all_numeric_predictors())

# Prep + bake
water_prep_sqrt <- prep(water_recipe_sqrt, training = water)
data_1_sqrt <- bake(water_prep_sqrt, new_data = NULL)

# Check skewness again
skewness_1_sqrt<- data_1_sqrt %>%
  select(-Potability) %>%
  summarise(across(everything(), ~ e1071::skewness(., na.rm = TRUE))) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Skewness")

skewness_1_sqrt

#visualizing this data

data_1_sqrt %>%
  pivot_longer(-Potability, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "lightblue", color = "black") +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distributions After Square Root Transformation & Normalization")

#lets do before vs after plot for solids
solids_compare <- bind_rows(
  tibble(Value = water$Solids, Stage = "Before"),
  tibble(Value = data_1_sqrt$Solids, Stage = "After")
)

ggplot(solids_compare, aes(x = Value, fill = Stage)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity", color = "black") +
  facet_wrap(~Stage, scales = "free") +
  theme_minimal() +
  labs(title = "Before vs After Transformation: Solids",
       x = "Solids", y = "Count") +
  scale_fill_manual(values = c("skyblue", "lightgreen"))

#yes now solids is slightly  symmetric. which was clear from skewness values for solids.



#1-D
class_prop
#as we have seen before there is a class imbalance with 61% being not portable(0) and 39% being portable(1)

library(themis) 
#first we need to convert portability into factor to use upsample step.
water <- water %>%
  mutate(Potability = as.factor(Potability))

# Final  recipe
water_recipe_1_D <- recipe(Potability ~ ., data = water) %>%
  step_impute_median(all_numeric_predictors()) %>%   
  step_sqrt(Solids) %>%                              
  step_normalize(all_numeric_predictors()) %>%       
  step_upsample(Potability)                          

# Prep & bake
water_prep_1_D <- prep(water_recipe_1_D, training = water)
data_1_D <- bake(water_prep_1_D, new_data = NULL)
head(data_1_D)
  
# new class distribution
table(data_1_D$Potability) %>% prop.table()


#In water safety, the cost of false negatives is high that is we want good sensitivity for Potable.
#so we used upsampling which duplicates minority class as there is a class imbalance.
#summarizing everything, 
#To address the moderate class imbalance potability was upsampled, balancing both classes at 50% each.
#The final preprocessing recipe applies median imputation (for missing values), a square root transformation for Solids(as it was skewed).
#normalization of all numerical predictors, and class balancing.
#This ensures the dataset is complete , standardized, and unbiased for all subsequent modeling tasks
#minimizing the risk of false negatives in water safety classification.


#@# --END OF ANSWER_1--
#@#=================================
#@# --START OF QUESTION_2--
#@# Title: Logistic Regression Classification with Optimal Threshold Selection
#@# You are implementing a logistic regression model to classify water potability and need to optimize the classification threshold for practical deployment.
#@#
#@# A) Using the tidymodels framework, create a workflow that includes a logistic regression model specification and 
#@#    a recipe for predicting water potability using all available chemical and physical parameters. 
#@#    Split your preprocessed data into training (75%) and testing (25%) sets with stratified sampling.
#@#
#@# B) Implement 5-fold cross-validation with stratified sampling to evaluate model performance. 
#@#    Calculate cross-validated accuracy, ROC AUC, sensitivity, specificity, and F1-score.
#@#    Explain why stratified sampling is particularly important for this water safety classification problem.
#@#
#@# C) Generate ROC curves and precision-recall curves for your model. 
#@#    Test different classification thresholds (0.3, 0.4, 0.5, 0.6, 0.7) and calculate the F1-score for each.
#@#    Identify the threshold that achieves the highest F1-score and extract the corresponding TPR and FPR values.
#@#
#@# D) Evaluate whether the threshold with highest F1-score is appropriate for water safety applications.
#@#    Discuss the trade-offs between false positives (declaring safe water unsafe) and false negatives (declaring unsafe water safe).
#@#    Compare your optimal threshold's performance on the test set and provide final model recommendations.
#@# --END OF QUESTION_2--
#@# --START OF ANSWER_2--
#@# PASTE R CODE BELOW HERE

#2-A
set.seed(123)

#for easy access 
water_final <- data_1_D #as this is the data we are going to use
head(water_final)

#lets split data
data_split <- initial_split(water_final, prop = 0.75, strata = Potability)
train_data <- training(data_split)
test_data  <- testing(data_split)

# recipe and spec for logistic model
log_recipe <- recipe(Potability ~ ., data = train_data)
log_spec <- logistic_reg(mode = "classification") %>%
  set_engine("glm")
#dont need to add any recipe steps as i am using the data which is completely preprocessed

# Workflow
log_workflow <- workflow() %>%
  add_model(log_spec) %>%
  add_recipe(log_recipe)

#now fitting the model
log_fit <- fit(log_workflow, data = train_data)

log_fit
#AIC value is 4164.(for refrence)



#2-B

set.seed(124)

# 5-fold cross-validation with stratification
water_folds <- vfold_cv(train_data, v = 5, strata = Potability)


# Cross-validation results
log_results_2_B <- fit_resamples(
  log_workflow,
  resamples = water_folds,
  metrics = metric_set(yardstick::accuracy, 
                       yardstick::roc_auc, 
                       yardstick::sensitivity, 
                       yardstick::specificity, 
                       yardstick::f_meas),
  control = control_resamples(save_pred = TRUE)
)

# Collect metrics
water_metrics <- collect_metrics(log_results_2_B)
water_metrics
#the results are bad in my opinion
#accuracy of 0.506 is basically like a coin flip, which is also clear from roc_auc being 0.504

#Stratified sampling is important because without stratification
#some training or validation folds could end up with too few potable samples, or even none in extreme random splits
#that would make performance metrics unreliable.
#according to me false positives are more dangerous.

#FP is unsafe water declared safe
#FN is safe water declared unsafe
#positive class is that water is potable.
#2-C

log_preds <- collect_predictions(log_results_2_B)
log_preds

# ROC Curve
roc_curve_data <- roc_curve(log_preds, truth = Potability, .pred_1)
autoplot(roc_curve_data) +
  ggtitle("ROC Curve: Logistic Regression")

#roc curve is almost near the diagnol line which is bad, this matches the values i got before.

# Precision-Recall Curve
pr_curve_data <- pr_curve(log_preds, truth = Potability, .pred_1)
autoplot(pr_curve_data) +
  ggtitle("Precision-Recall Curve: Logistic Regression")

#generally in class imbalance situations pr curve is more informative  than roc curve.
#according to me
#High recall = fewer false negatives (unsafe water missed)
#High precision = fewer false positives (declaring safe water unsafe)
#and we need a balnce between this which is F1 score.

#according to me we can't declare unsafe water safe as it has more issues than the other way round
#so if given a choice between recall and precision I would choose higher recall.
#don't get me wrong, high precison is also required as resouce would be wasted if we declare safe water unsafe in few regions.
#but the damage cause by false negatives is higher in my opinion


# testing differentt  thresholds
thresholds <- c(0.3, 0.4, 0.5, 0.6, 0.7)

threshold_metrics <- map_dfr(thresholds, function(t) {
  preds <- log_preds %>%
    mutate(.pred_class = ifelse(.pred_1 >= t, "1", "0") %>% factor(levels = levels(Potability)))
  
  tibble(
    Threshold = t,
    F1 = f_meas_vec(preds$Potability, preds$.pred_class, event_level = "second"),
    Sensitivity = sensitivity_vec(preds$Potability, preds$.pred_class, event_level = "second"),
    Specificity = specificity_vec(preds$Potability, preds$.pred_class, event_level = "second"),
    Accuracy = accuracy_vec(preds$Potability, preds$.pred_class)
  )
})

threshold_metrics
#at lower thresholds no or very few false negatives and as we go higher sensitivity decreases and precision increases.

#Highest F1 score is at 0.3 with a value of 0.667.
#TPR(sensitivity)=1.
#specificity=0
#FPR= 1-specificity= 1-0=1.

#2-D

#ok now this is based on opinions.
#if i was a health officer even at the cost of massive resouce wastage on false positives, i cannot have false positives as it risky for heath.
# so i would say threshold of 0.3(Highest F1 score) is not optimal as there will be 0 FN cases but massive FP cases.
# again if the operation(resouces) costs are way too severe then I dont know.
#from a safety officer viewpont threshold should be 0.7 or 0.6.
#from a buisiness viepoint for a buisiness man it can be 0.5 ,if the operational costs for all FP cases is greater than medical cost of FN cases for other thresholds.
#Here I am going from safety office viewpoint,
#again there are 2 problems, choosing between spec =1 and 0 sens or spec=0.999 with slighly better sens.
#as a health officer I cannot let non-potable water be labelled as potable even if it is one case so optimal thresh should be 0.7.

test_preds <- predict(log_fit, test_data, type = "prob") %>%
  bind_cols(test_data %>% select(Potability))

optimal_threshold <- 0.7
test_preds_opt <- test_preds %>%
  mutate(.pred_class = ifelse(.pred_1 >= optimal_threshold, "1", "0") %>%
           factor(levels = levels(Potability)))


test_preds_def <- test_preds %>%
  mutate(.pred_class = ifelse(.pred_1 >= 0.5, "1", "0") %>%
           factor(levels = levels(Potability)))
test_preds_def

#test 1

# Optimal threshold (0.7)
test_metrics_opt <- tibble(
  Threshold = 0.7,
  Accuracy = accuracy_vec(test_preds_opt$Potability, test_preds_opt$.pred_class),
  F1 = f_meas_vec(test_preds_opt$Potability, test_preds_opt$.pred_class, event_level = "second"),
  Sensitivity = sensitivity_vec(test_preds_opt$Potability, test_preds_opt$.pred_class, event_level = "second"),
  Specificity = specificity_vec(test_preds_opt$Potability, test_preds_opt$.pred_class, event_level = "second"),
  Precision = precision_vec(test_preds_opt$Potability, test_preds_opt$.pred_class, event_level = "second")
)

# Default threshold for comparision
test_metrics_def <- tibble(
  Threshold = 0.5,
  Accuracy = accuracy_vec(test_preds_def$Potability, test_preds_def$.pred_class),
  F1 = f_meas_vec(test_preds_def$Potability, test_preds_def$.pred_class, event_level = "second"),
  Sensitivity = sensitivity_vec(test_preds_def$Potability, test_preds_def$.pred_class, event_level = "second"),
  Specificity = specificity_vec(test_preds_def$Potability, test_preds_def$.pred_class, event_level = "second"),
  Precision = precision_vec(test_preds_def$Potability, test_preds_def$.pred_class, event_level = "second")
)

test_comparison_1 <- bind_rows(test_metrics_def, test_metrics_opt)
test_comparison_1

#test 2
train_metrics_opt <- threshold_metrics %>%
  filter(Threshold == 0.7) %>%
  mutate(Source = "Train ")
#mutated to know which is which
test_metrics_opt <- test_metrics_opt %>%
  mutate(Source = "Test")

test_comparison_2<- bind_rows(train_metrics_opt, test_metrics_opt)
test_comparison_2

#the question did not specify what to compare with so I compared with default model(thresh=0.5), and with train data set.

#the train and test for optimal thresh gives similar metrics as at thresh of 0.7 the model almost classifies everything as not potable.
#and with default as expected thresh of 0.7 has lower sens and F1 but higher spec and precision.

conf_matrix_2 <- conf_mat(test_preds_opt, truth = Potability, estimate = .pred_class)
conf_matrix_2

# just for better look
autoplot(conf_matrix_2, type = "heatmap") +
  ggtitle("Confusion Matrix at Threshold 0.7")



#is logistic model really correct for this dataset?



#@# --END OF ANSWER_2--
#@#=================================
#@# --START OF QUESTION_3--
#@# Title: Decision Tree Classification and Overfitting Detection
#@# You are exploring decision tree models for water potability classification to provide interpretable rules for water quality assessment while learning to detect overfitting.
#@#
#@# A) Using tidymodels, create a decision tree workflow for predicting water potability. 
#@#    Set min_n = 20 (minimum observations per node) and keep this fixed throughout the analysis.
#@#    Split your preprocessed data into training (75%) and testing (25%) sets with stratified sampling.
#@#
#@# B) Train decision tree models with different tree depths: e.g., 3, 5, 10, 20 (or your own choices)
#@#    For each tree depth, evaluate the model performance on both the training set and test set.
#@#    Calculate accuracy for both training and test sets and create a table showing the results for all tree depths.
#@#
#@# C) Create a visualization (line plot or bar chart) showing how training accuracy and test accuracy change as tree depth increases. 
#@#    Analyze the patterns you observe and identify at what tree depth overfitting begins to occur.
#@#    Explain the relationship between model complexity (tree depth) and the gap between training and test performance.
#@#
#@# D) Select the optimal tree depth based on your overfitting analysis and fit the final decision tree model. 
#@#    Create a visualization of the tree structure and interpret the decision rules.
#@#    Evaluate this final model using multiple metrics (accuracy, sensitivity, specificity, F1-score, ROC AUC) and explain how the tree makes decisions about water potability in terms that water quality managers could understand.
#@# --END OF QUESTION_3--
#@# --START OF ANSWER_3--
#@# PASTE R CODE BELOW HERE

#3-A
library(rpart.plot)

set.seed(324)
#lets use a diffrent split 
data_split_3A <- initial_split(water_final, prop = 0.75, strata = Potability)
train_data_3A <- training(data_split_3A)
test_data_3A  <- testing(data_split_3A)

# Recipe (no extra steps since already preprocessed)
tree_recipe <- recipe(Potability ~ ., data = train_data_3A)

# Decision Tree model spec
tree_spec <- decision_tree(
  tree_depth = tune(),   # will tune depth in 3B
  min_n = 20             # fixed as per the question
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# Workflow
tree_workflow <- workflow() %>%
  add_model(tree_spec) %>%
  add_recipe(tree_recipe)

tree_workflow

#3-B
set.seed(458)
# Depths to test
depths <- c(3, 5, 10, 20)

#  evaluate a decision tree at a given depth
evaluate_tree_3B <- function(depth) {
  
  spec <- decision_tree(
    tree_depth = depth,
    min_n = 20
  ) %>%
    set_engine("rpart") %>%
    set_mode("classification")
  
  wf <- workflow() %>%
    add_model(spec) %>%
    add_recipe(tree_recipe)   # from 3A
  
  fit <- fit(wf, data = train_data_3A)
  
  # Predictions
  train_preds_3_B <- predict(fit, train_data_3A) %>%
    bind_cols(train_data_3A %>% select(Potability))
  
  test_preds_3_B <- predict(fit, test_data_3A) %>%
    bind_cols(test_data_3A %>% select(Potability))
  
  # Metrics
  tibble(
    Tree_Depth = depth,
    Train_Accuracy = accuracy_vec(train_preds_3_B$Potability, train_preds_3_B$.pred_class),
    Test_Accuracy  = accuracy_vec(test_preds_3_B$Potability, test_preds_3_B$.pred_class)
  )
}


# Evaluating all depths 
tree_results_3B <- map_dfr(depths, evaluate_tree_3B)
tree_results_3B


#3-C

ggplot(tree_results_3B, aes(x = Tree_Depth)) +
  geom_line(aes(y = Train_Accuracy, color = "Train")) +
  geom_line(aes(y = Test_Accuracy, color = "Test")) +
  geom_point(aes(y = Train_Accuracy, color = "Train")) +
  geom_point(aes(y = Test_Accuracy, color = "Test")) +
  labs(title = "Decision Tree Accuracy vs Depth",
       y = "Accuracy", x = "Tree Depth") +
  theme_minimal() 

#there is wideninig gap whichstarts at depth=10. So overfitting mostly likely starts from depth =10.


#Increasing depth means increasing complexity, allowing it to capture more patterns in the training data SO Training accuracy keeps improving 
#t first, both training and test accuracy rise, since the model is learning useful structure. However, beyond a certain point, additional depth mostly captures noise or dataset-specific quirks..
#This causes training accuracy to keep climbing while test accuracy stalls or declines, creating a widening train–test gap
#that is the start of overfitting

#3-D
#optimal value is 10  because that’s where test accuracy peaks before falling down
set.seed(997)
final_tree_spec_pruned <- decision_tree(tree_depth = 10, min_n = 20) %>%
  set_engine("rpart") %>%
  set_mode("classification")

final_tree_fit_pruned <- workflow() %>%
  add_model(final_tree_spec_pruned) %>%
  add_recipe(tree_recipe) %>%
  fit(data = train_data_3A)

# Predictions (class)
test_preds_pruned_class <- predict(final_tree_fit_pruned, test_data_3A, type = "class") %>%
  bind_cols(test_data_3A %>% select(Potability))

# Predictions (probability)
test_preds_pruned_prob <- predict(final_tree_fit_pruned, test_data_3A, type = "prob") %>%
  bind_cols(test_data_3A %>% select(Potability))

# Metrics for pruned tree
metrics_pruned <- tibble(
  Model       = "Pruned Tree",
  Accuracy    = accuracy_vec(test_preds_pruned_class$Potability, test_preds_pruned_class$.pred_class),
  Sensitivity = sensitivity_vec(test_preds_pruned_class$Potability, test_preds_pruned_class$.pred_class, event_level = "second"),
  Specificity = specificity_vec(test_preds_pruned_class$Potability, test_preds_pruned_class$.pred_class, event_level = "second"),
  F1          = f_meas_vec(test_preds_pruned_class$Potability, test_preds_pruned_class$.pred_class, event_level = "second"),
  ROC_AUC     = roc_auc(test_preds_pruned_prob, truth = Potability, .pred_1)$.estimate
)
metrics_pruned

#the tree is balaned but it is not protecting against any type of error.
#The roc_auc value is low.
#this model is simple and interpretable but has mediocre performance
# Final pruned Tree
final_tree_model_pruned <- extract_fit_parsnip(final_tree_fit_pruned)$fit
rpart.plot(final_tree_model_pruned, type = 3, fallen.leaves = TRUE, cex = 0.7, main = "Decision Tree (Pruned)")




#unpruned trees have higher risk for overfitting 


#looks like sulphate is the most important predictor  
#if sulphate is high then ph and hardness come into action or else they dont.

#The decision tree works like a flowchart. At each split, it asks a yes/no question about a specific water quality factor. 
#The path you follow through the tree eventually leads to a prediction: potable (1) or not potable (0)
#Sulfate Levels are Checked First(If sulfate is very low (< -1.4, normalized scale), the tree looks at pH),
#pH is a Frequent Decider(Very high pH (> 2) is associated with Not Potable outcomes)
#Hardness Adds an Extra Safety Filter(If Hardness is below 2.6 while sulfates are moderate, it often indicates Not Potable)
#Organic Carbon and Chloramines are bad and few aditional splits which are clear from the tree


#Very high pH values (> 2) usually signal unsafe water, while moderate hardness levels act as an extra safety check. 
#If sulfates are moderate, organic carbon and chloramines levels come into play
#In summary, sulfate acts as the gatekeeper, pH is a frequent decider, and hardness, organic carbon, and chloramines are supporting checks to refine the decision


#Suppose a water sample has Sulfate of 3 and ph of 2 then it is not potable according to the decison tree.

#@# --END OF ANSWER_3--
#@#=================================
#@# --START OF QUESTION_4--
#@# Title: Ensemble Methods - Random Forest and XGBoost Classification
#@# You are implementing advanced ensemble methods to improve water potability prediction accuracy while maintaining robust performance.
#@#
#@# A) Using tidymodels, create workflows for both Random Forest and XGBoost models for water potability classification.
#@#    For Random Forest, set up tuning for only the number of trees (trees parameter), keeping mtry at its default value.
#@#    For XGBoost, set up tuning for only the number of trees (trees parameter), keeping learn_rate = 0.3 and other parameters at default values.
#@#    Split your preprocessed data into training (75%) and testing (25%) sets with stratified sampling.
#@#
#@# B) Implement cross-validation tuning for both models ensuring no data leakage occurs. 
#@#    For both Random Forest and XGBoost: test trees = (10, 100, 200).
#@#    Use F1-score as your tuning metric and identify the best number of trees for each model.
#@#
#@# C) Fit the final tuned models for both Random Forest and XGBoost using the best number of trees identified in part B.
#@#    Evaluate both models on the test set and compare their cross-validation performance to their test set performance.
#@#    Discuss any differences between CV and test performance and what this indicates about model generalization.
#@#
#@# D) Evaluate both ensemble models on the test set and compare their performance to each other.
#@#    Discuss the benefits and potential drawbacks of using ensemble methods for water quality assessment, and which of the two ensemble approaches performs better for this specific problem.
#@# --END OF QUESTION_4--
#@# --START OF ANSWER_4--
#@# PASTE R CODE BELOW HERE

library(xgboost)
library(ranger)  


#4-A

set.seed(111)

# splitting data again
data_split_4A <- initial_split(water_final, prop = 0.75, strata = Potability)
train_data_4A <- training(data_split_4A)
test_data_4A  <- testing(data_split_4A)

# Recipe (no extra steps since already preprocessed)
rf_xgb_recipe <- recipe(Potability ~ ., data = train_data_4A)

# Random Forest specification 
rf_spec <- rand_forest(
  trees = tune()           
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Workflow for RF
rf_workflow <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(rf_xgb_recipe)

# XGBoost specification 
xgb_spec <- boost_tree(
  trees = tune(),
  learn_rate = 0.3     
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Workflow for XGBoost
xgb_workflow <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(rf_xgb_recipe)


rf_workflow
xgb_workflow


#4-B

set.seed(222)

# Creating folds
cv_folds_4B <- vfold_cv(train_data_4A, v = 5, strata = Potability)

# Grid of trees to test
tree_grid <- tibble(trees = c(10, 100, 200))

# Random Forest tuning
rf_tune_results <- tune_grid(
  rf_workflow,
  resamples = cv_folds_4B,
  grid = tree_grid,
  metrics = metric_set(f_meas),
  control = control_grid(save_pred = TRUE)
)

# Collect RF results
rf_metrics <- collect_metrics(rf_tune_results)
rf_metrics

# Best RF based on F1
best_rf <- select_best(rf_tune_results, metric = "f_meas")
best_rf

# XGBoost tuning
xgb_tune_results <- tune_grid(
  xgb_workflow,
  resamples = cv_folds_4B,
  grid = tree_grid,
  metrics = metric_set(f_meas),
  control = control_grid(save_pred = TRUE)
)

# XGB results
xgb_metrics <- collect_metrics(xgb_tune_results)
xgb_metrics

# Best XGB based on F1
best_xgb <- select_best(xgb_tune_results, metric = "f_meas")
best_xgb

#ensured that there is no data leakage.

#4-C

set.seed(333)

# Final specs with best number of trees (100)
final_rf_spec <- rand_forest(
  trees = best_rf$trees 
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

final_xgb_spec <- boost_tree(
  trees = best_xgb$trees, 
  learn_rate = 0.3
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Workflows
final_rf_workflow <- workflow() %>%
  add_model(final_rf_spec) %>%
  add_recipe(rf_xgb_recipe)

final_xgb_workflow <- workflow() %>%
  add_model(final_xgb_spec) %>%
  add_recipe(rf_xgb_recipe)

# Fit on training data
final_rf_fit <- fit(final_rf_workflow, data = train_data_4A)
final_xgb_fit <- fit(final_xgb_workflow, data = train_data_4A)

# Predictions on test set
rf_preds <- predict(final_rf_fit, test_data_4A, type = "prob") %>%
  bind_cols(predict(final_rf_fit, test_data_4A)) %>%
  bind_cols(test_data_4A %>% select(Potability))

xgb_preds <- predict(final_xgb_fit, test_data_4A, type = "prob") %>%
  bind_cols(predict(final_xgb_fit, test_data_4A)) %>%
  bind_cols(test_data_4A %>% select(Potability))

# Compute metrics for both models
rf_metrics_test <- tibble(
  Model       = "Random Forest",
  Accuracy    = accuracy_vec(rf_preds$Potability, rf_preds$.pred_class),
  Sensitivity = sensitivity_vec(rf_preds$Potability, rf_preds$.pred_class, event_level = "second"),
  Specificity = specificity_vec(rf_preds$Potability, rf_preds$.pred_class, event_level = "second"),
  F1          = f_meas_vec(rf_preds$Potability, rf_preds$.pred_class, event_level = "second"),
  ROC_AUC     = roc_auc(rf_preds, truth = Potability, .pred_1)$.estimate
)

xgb_metrics_test <- tibble(
  Model       = "XGBoost",
  Accuracy    = accuracy_vec(xgb_preds$Potability, xgb_preds$.pred_class),
  Sensitivity = sensitivity_vec(xgb_preds$Potability, xgb_preds$.pred_class, event_level = "second"),
  Specificity = specificity_vec(xgb_preds$Potability, xgb_preds$.pred_class, event_level = "second"),
  F1          = f_meas_vec(xgb_preds$Potability, xgb_preds$.pred_class, event_level = "second"),
  ROC_AUC     = roc_auc(xgb_preds, truth = Potability, .pred_1)$.estimate
)

# Combining results
test_comparison_4C <- bind_rows(rf_metrics_test, xgb_metrics_test)
test_comparison_4C

#the resuts are similar to each other with slight differences but randomn forest has specificity and  F1 score.


#Random Forest is preferable for balanced performance and higher specificity.

# Compare CV metrics with test metrics
rf_cv_metrics <- show_best(rf_tune_results, metric = "f_meas", n = 1)
xgb_cv_metrics <- show_best(xgb_tune_results, metric = "f_meas", n = 1)

rf_cv_metrics
xgb_cv_metrics

#The small gap between CV and test metrics shows both models have good generalization and their cross-validation F1 scores were close to test results.
#good thing is that neither Random Forest nor XGBoost suffered a major performance drop when moving from CV to the test set.
#I feel like these models are robust for real world usage.

#4-D

test_comparison_4C

#Ensemble methods like Random Forest and XGBoost combine multiple models to improve prediction accuracy and robustness
#that reduces the risk of overfitting by averaging multiple weak learners
#They can catch complex parterns and genearlize well.

#But they are complex and computationally expensive
#Tuning and validation require more effort to avoid overfitting or data leakage

#i would choose randomn forest over XGBoost for it's higher specificity and F1 score which are very important in this case.



#@# --END OF ANSWER_4--
#@#=================================
#@# --START OF QUESTION_5--
#@# Title: Support Vector Machine Classification with Kernel Tuning
#@# You are exploring Support Vector Machine models with different kernels to find the optimal decision boundary for water potability classification.
#@#
#@# A) Using tidymodels, create SVM workflows with three different kernels: linear, polynomial, and radial basis function (RBF).
#@#    Set up hyperparameter tuning for only the cost parameter (regularization strength) for all three kernel types.
#@#    For polynomial kernel, fix degree = 3. For RBF kernel, fix rbf_sigma = 0.01. Linear kernel has no additional parameters.
#@#    Split your preprocessed data into training (60%), validation (20%), and testing (20%) sets with stratified sampling.
#@#
#@# B) Use the validation set created in Part A for tuning all three SVM models. 
#@#    For all kernels: test cost = (0.1, 1, 10, 100).
#@#    Use precision as your tuning metric on the validation set and identify the best cost parameter for each kernel type.
#@#
#@# C) Compare the performance of all three tuned SVM models on the test set.
#@#    Analyze which kernel type performs best and discuss why it might be most suitable for this water potability dataset.
#@#    (Consider the complexity of decision boundaries that each kernel can create).
#@#
#@# D) Considering all the different approaches you tried in the 5 problems of the Lab, and
#@#    provide a well-justified recommendation for water potability classification including model choice, 
#@#    key performance metrics, and practical considerations for deployment.
#@# --END OF QUESTION_5--
#@# --START OF ANSWER_5--
#@# PASTE R CODE BELOW HERE

#5-A
set.seed(444)

#splitting data
data_split_5A <- initial_split(water_final, prop = 0.6, strata = Potability)
train_data_5A <- training(data_split_5A)
data_5_A      <- testing(data_split_5A)

# Second split: 50/50 of the remaining 40% to 20% validation, 20% test
valid_test_split_5A <- initial_split(data_5_A, prop = 0.5, strata = Potability)
valid_data_5A <- training(valid_test_split_5A)  
test_data_5A  <- testing(valid_test_split_5A)   

# Check proportions
nrow(train_data_5A) / nrow(water_final)  
nrow(valid_data_5A) / nrow(water_final)  
nrow(test_data_5A) / nrow(water_final)   
# Recipe (no extra preprocessing since already done before)
svm_recipe <- recipe(Potability ~ ., data = train_data_5A)

#Linear kernel SVM
svm_linear_spec <- svm_linear(
  cost = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_linear_wf <- workflow() %>%
  add_model(svm_linear_spec) %>%
  add_recipe(svm_recipe)

# Polynomial kernel SVM (degree = 3 fixed)
svm_poly_spec <- svm_poly(
  cost = tune(),
  degree = 3
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_poly_wf <- workflow() %>%
  add_model(svm_poly_spec) %>%
  add_recipe(svm_recipe)

# RBF kernel SVM (sigma fixed at 0.01)
svm_rbf_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = 0.01
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_rbf_wf <- workflow() %>%
  add_model(svm_rbf_spec) %>%
  add_recipe(svm_recipe)


svm_linear_wf
svm_poly_wf
svm_rbf_wf

#5-B

set.seed(555)

# Grid of cost values
cost_grid <- tibble(cost = c(0.1, 1, 10, 100))

#using  Helper function
tune_svm_on_val <- function(workflow, kernel_name, train_data, valid_data, cost_values) {
  
  results <- map_dfr(cost_values, function(c) {
    
    # Update the workflow with the new cost parameter
    wf <- workflow %>%
      update_model(
        extract_spec_parsnip(workflow) %>% 
          set_args(cost = c)
      )
    
    # Fit on training data
    fit <- fit(wf, data = train_data)
    
    # Predict on validation data
    preds <- predict(fit, valid_data, type = "class") %>%
      bind_cols(valid_data %>% select(Potability))
    
    # Compute precision
    tibble(
      Kernel = kernel_name,
      Cost = c,
      Precision = precision_vec(preds$Potability, preds$.pred_class, event_level = "second")
    )
  })
  
  return(results)
}

# Run tuning for each kernel
svm_linear_results <- tune_svm_on_val(svm_linear_wf, "Linear", train_data_5A, valid_data_5A, cost_grid)
svm_poly_results   <- tune_svm_on_val(svm_poly_wf,   "Polynomial", train_data_5A, valid_data_5A, cost_grid)
svm_rbf_results    <- tune_svm_on_val(svm_rbf_wf,    "RBF", train_data_5A, valid_data_5A, cost_grid)

# Combine results
svm_results_5B <- bind_rows(svm_linear_results, svm_poly_results, svm_rbf_results)

# Best cost per kernel
best_costs_5B <- svm_results_5B %>%
  group_by(Kernel) %>%
  slice_max(Precision, with_ties = FALSE)

# View results
svm_results_5B
best_costs_5B


#just for visualization

ggplot(svm_results_5B, aes(x = factor(Cost), y = Precision, color = Kernel, group = Kernel)) +
  geom_line() +
  geom_point(size = 3) +
  labs(title = "Precision by Kernel and Cost",
       x = "Cost Parameter", y = "Precision") +
  theme_minimal()

#5-C

set.seed(666)

evaluate_svm_on_test <- function(workflow, best_cost, kernel_name, train_data, test_data) {
  
  # Update workflow with best cost
wf <- workflow %>%
    update_model(
      extract_spec_parsnip(workflow) %>% 
        set_args(cost = best_cost)
    )
  
# Fit on training data
fit <- fit(wf, data = train_data)
  
# Predictions
preds_class <- predict(fit, test_data, type = "class") %>%
    bind_cols(test_data %>% select(Potability))
  
preds_prob <- predict(fit, test_data, type = "prob") %>%
    bind_cols(test_data %>% select(Potability))
  

tibble(
    Kernel      = kernel_name,
    Cost        = best_cost,
    Accuracy    = accuracy_vec(preds_class$Potability, preds_class$.pred_class),
    Sensitivity = sensitivity_vec(preds_class$Potability, preds_class$.pred_class, event_level = "second"),
    Specificity = specificity_vec(preds_class$Potability, preds_class$.pred_class, event_level = "second"),
    Precision   = precision_vec(preds_class$Potability, preds_class$.pred_class, event_level = "second"),
    F1          = f_meas_vec(preds_class$Potability, preds_class$.pred_class, event_level = "second"),
    ROC_AUC     = roc_auc(preds_prob, truth = Potability, .pred_1)$.estimate
  )
}

# Run evaluation for each kernel
svm_test_results <- bind_rows(
  evaluate_svm_on_test(svm_linear_wf, best_costs_5B %>% filter(Kernel=="Linear") %>% pull(Cost), "Linear", train_data_5A, test_data_5A),
  evaluate_svm_on_test(svm_poly_wf,   best_costs_5B %>% filter(Kernel=="Polynomial") %>% pull(Cost), "Polynomial", train_data_5A, test_data_5A),
  evaluate_svm_on_test(svm_rbf_wf,    best_costs_5B %>% filter(Kernel=="RBF") %>% pull(Cost), "RBF", train_data_5A, test_data_5A)
)

svm_test_results

# Compare Validation vs Test Precision
validation_vs_test <- best_costs_5B %>%
  rename(Validation_Precision = Precision, Best_Cost = Cost) %>%
  left_join(svm_test_results %>% select(Kernel, Precision), by = "Kernel") %>%
  rename(Test_Precision = Precision)

validation_vs_test

#I think it should be rbf as it has a slight edge in test precision
#polynomial is good too but rbf genrailzes a bit better in this case.

#5-D
#I would say a randomn forest model is the best for this case as it has good metrics which are required.
test_comparison_4C
#It has high specifivity which is very crucial in this case with good F1 score ,sensitivity and accuracy too.
#it's metrics are clearly better than other models(XGboost is very close to randomn forest)
#This makes it highly reliable in minimizing both false negatives  and false positives  which is critical for public health decisions
# Random Forest are robust to noise and imbalanced datasets, reducing the chance of erratic predictions
#It is computationally efficient and scales well.
#Overall, Random Forest strikes the right balance between robustness, computational efficiency, and predictive power
#and its feature importance scores can provide actionable insights for water quality managers.



#@# --END OF ANSWER_5--
#@#=================================