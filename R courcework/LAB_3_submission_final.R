#@# Statistical Analysis Lab Submission Template
#@# Instructions:
#@# 1. Do not modify any line beginning with #@#
#@# 2. Do not begin any line in your solution programs or comments with #@#
#@# 3. Paste your R code below the line PASTE R CODE BELOW HERE for each answer.
#@# 4. Ensure each answer corresponds to the correct question number.
#@# 5. Provide code for all 5 questions, even if incomplete, to avoid parsing errors.
#@# 6. Save this file as a plain .txt file and upload via the Google Form.
#@#=================================
#@# --START OF QUESTION_1--
#@# Title: Automotive Performance Analysis (mtcars dataset)
#@# You are a data analyst for an automotive magazine investigating fuel efficiency and performance characteristics of 1970s cars.
#@#
#@# A) Examine the distribution of miles per gallon (mpg) in the dataset. Calculate appropriate descriptive statistics and assess whether the distribution appears to follow a normal distribution. Based on your analysis, determine if there are any outliers and comment on the shape of the distribution.
#@#
#@# B) The horsepower variable (hp) appears to be right-skewed. Apply an appropriate transformation to make this variable more normally distributed. Compare the skewness before and after transformation and create visualizations to demonstrate the improvement.
#@#
#@# C) Calculate a 95% confidence interval for the true mean fuel efficiency (mpg) of all cars represented in this dataset. Interpret what this interval means in practical terms.
#@#
#@# D) Test whether there is a significant difference in fuel efficiency between automatic and manual transmission cars (am variable, where 0 = automatic, 1 = manual). Calculate the effect size and determine if the difference is practically meaningful.
#@# --END OF QUESTION_1--
#@# --START OF ANSWER_1--
#@# PASTE R CODE BELOW HERE


# Install required packages (run this line ONCE)
install.packages(c("tidyverse", "palmerpenguins", "gapminder", "moments", "car", "pwr", "effsize", "vcd"))

# Load required libraries
library(tidyverse)
library(palmerpenguins)
library(gapminder)
library(moments)
library(car)
library(pwr)
library(effsize)
library(vcd)

#1-A
summary(mtcars$mpg)
sd(mtcars$mpg)  
skewness(mtcars$mpg)  
kurtosis(mtcars$mpg)  

#shape of mtcars
dim(mtcars)
#checking for NA values to decide to clean or not clean dataset
sum(is.na(mtcars))        
colSums(is.na(mtcars))
#as there are no NA values cleaning is not needed

# Histogram 
ggplot(mtcars, aes(x = mpg)) +
  geom_histogram(binwidth = 2, fill = "lightblue", color = "black") +
  labs(title = "Distribution of MPG", x = "mpg", y = "count")

#From the graph we understand that it is not perfectly normal distribution.(cannot use t-test without transformation)
#It’s slightly right-skewed (there's a longer tail on the left).
#Most cars seem clustered between 13–22 mpg.
#Can have outliers for mpg>30
# We can say its  right-skewed from summary as Mean is greater than median(20.09>19.20)
#skewness is 0.64 and kurtosis is less than 3 (2.799).

#Q-Q plot to check distribution
qqnorm(mtcars$mpg)
qqline(mtcars$mpg)

#from the graph we understand that few points lie roughly along the diagonal line.
#there’s a noticeable upward curve in the top-right corner:
#the bottom-left corner dips a bit below the line
#It is not symmetric.


# Checking for  outliers using IQR method
Q1 <- quantile(mtcars$mpg, 0.25)
Q3 <- quantile(mtcars$mpg, 0.75)
IQR_mpg <- Q3 - Q1

lower_bound <- Q1 - 1.5 * IQR_mpg
upper_bound <- Q3 + 1.5 * IQR_mpg

outliers <- mtcars$mpg[mtcars$mpg < lower_bound | mtcars$mpg > upper_bound]
outliers
#There is one outlier
#row of that outlier
mtcars %>%
  filter(mpg %in% outliers)



#1-B

# Applying Log transformation (As it is right skewed)
mtcars_log <- data.frame(mtcars) %>%
  mutate(
    log_transformation = log(hp)
     )
mtcars$log_transformation
# Skewness after transformation
skewness(mtcars_log$log_transformation)
#new value is -0.0904 and before transformation skewness was 0.64
#this means that our data transformed from right-skewed to almost symmetric.

# Histogram after log transformation
ggplot(mtcars, aes(x = log_hp)) +
  geom_histogram(binwidth = 0.1, fill = "lightgreen", color = "black") +
  labs(title = "after_log", x = "log_hp")



#1-C
#just using simple one sample t-test 
t.test(mtcars$mpg, conf.level = 0.95)

#the 95% confidence interval is  (17.91768, 22.26357)

#1-D

# First I want to visualize data, and best way according to me might be box-plot
boxplot(mpg ~ am, data = mtcars,
        names = c("Automatic", "Manual"),
        main = "MPG by Transmission Type",
        ylab = "Miles Per Gallon",
        col = c("pink", "lightblue"))
#we can say that Manual cars generally have higher MPG but automatic car values are more consistent.

t_test_result <- t.test(mpg ~ am, data = mtcars)
t_test_result
#as p-value is less than 0.05 we reject null hypothesis that there is no significant difference in fuel effeciency.
#Which means that there is a  significant difference in mpg between automatic and manual cars.

# Calculate Cohen's d (effect size)
library(effsize)
cohen_d_result <- cohen.d(mtcars$mpg, mtcars$am)
cohen_d_result

#as magnitude of d is 4.60 there is significant difference.




#@# --END OF ANSWER_1--
#@#=================================
#@# --START OF QUESTION_2--
#@# Title: Antarctic Penguin Research (penguins dataset)
#@# You are a marine biologist studying penguin populations in Antarctica and need to analyze morphological differences between species.
#@#
#@# A) Test whether the mean body mass of all penguins in the dataset is significantly different from 4000 grams. State your hypotheses clearly and interpret your results.
#@#
#@# B) Compare the bill lengths between Adelie and Gentoo penguins. Determine if the assumptions for a parametric test are met, and conduct the appropriate statistical test.
#@#
#@# C) Using the same data from Part B, conduct a non-parametric test to verify your findings. Compare the results between the parametric and non-parametric approaches and explain any differences.
#@#
#@# D) Calculate the effect size for the difference in bill lengths between Adelie and Gentoo penguins. Determine whether the statistical significance translates to practical significance for penguin identification purposes.
#@# --END OF QUESTION_2--
#@# --START OF ANSWER_2--
#@# PASTE R CODE BELOW HERE

#shape of penguins dataset
dim(penguins)

sum(is.na(penguins))          
colSums(is.na(penguins))
#There are 5 columns with NA values


#First let's clean the dataset from NA values only for the required columns as we don't want to reduce our observations for unneded features
penguins_clean <- penguins %>% filter(!is.na(body_mass_g), !is.na(bill_length_mm)) #didn't do for species as species has no NA values
penguins_clean

#2-A
#one-sample t-test
t.test(penguins_clean$body_mass_g, mu = 4000) #H0 would be mu=4000 and H1 would be mu!=4000.
#assuming alpha as 0.5
#as p values is way less that 0.5 we reject the null hypothesis
#that means mean body mass is significantly different from 4000g


#2-B
#lets get the dataset we want first
penguins_want <- penguins_clean %>% 
  filter(species %in% c("Adelie", "Gentoo")) %>% 
  select(species, bill_length_mm) %>%
  na.omit()

summary(penguins_want)

#Q-Q plot for checking normality
# For Adelie
qqnorm(penguins_want$bill_length_mm[penguins_want$species == "Adelie"])
qqline(penguins_want$bill_length_mm[penguins_want$species == "Adelie"], col = "blue")

# For Gentoo
qqnorm(penguins_want$bill_length_mm[penguins_want$species == "Gentoo"])
qqline(penguins_want$bill_length_mm[penguins_want$species == "Gentoo"], col = "red")

#I feel both are close to normal distribution but I am not that sure because of the deviations
#So I want use shapiro test

shapiro.test(penguins_want$bill_length_mm[penguins_want$species == "Adelie"])
#p-value is greater than 0.05(0.7166) so it is following a normal distribution

shapiro.test(penguins_want$bill_length_mm[penguins_want$species == "Gentoo"])
#p-value is not greater than 0.05(0.01349) so it is not following a normal distribution


#we cannot use 2-sample t test but just using to compare with other test


#Test for equal Variance
leveneTest(bill_length_mm ~ species, data = penguins_want)#null hypothesis is that variances are equal
#as p value is greater than 0.05 (0.2009) we accept null hypothesis and we can say variances are equal


#as variances are equal
t.test(bill_length_mm ~ species, data = penguins_want, var.equal = TRUE)

#as p value is very less, that means 
#The difference in bill lengths between Adelie and Gentoo penguins is highly significant.


#2-C
#If we don't assume normality and do non-parametric test 
wilcox.test(bill_length_mm ~ species, data = penguins_want, exact = FALSE)
#as p value is significantly lower than 0.05 we reject null hypothesis.
#The difference in bill lengths between Adelie and Gentoo penguins is highly significant

#2-D
ade <- penguins_want %>% 
  filter(species=="Adelie")
gen <- penguins_want %>% 
  filter(species=="Gentoo") 

#did in this method because i got NAN valu even though the dataset was clean 
cohen.d(gen$bill_length_mm, ade$bill_length_mm, pooled=TRUE)

#as magnitude of d is 3.048 it means that
# the average Gentoo penguin has a bill length more than 3 standard deviations longer than the average Adelie penguin.
#high significance


#@# --END OF ANSWER_2--
#@#=================================
#@# --START OF QUESTION_3--
#@# Title: Vitamin C and Tooth Growth Study (ToothGrowth dataset)
#@# You are analyzing the results of a controlled experiment studying the effect of vitamin C on tooth growth in guinea pigs.
#@#
#@# A) Conduct a one-way ANOVA to test whether there are significant differences in tooth growth across all six treatment combinations (2 supplement types × 3 dose levels). Report the F-statistic, p-value, and overall conclusion.
#@#
#@# B) If your ANOVA in Part A was significant, conduct appropriate post-hoc tests to determine which specific treatment combinations differ significantly from each other. Summarize the key pairwise differences.
#@#
#@# C) Suppose you are planning a follow-up study and want to detect a medium effect size (Cohen's d = 0.5) with 80% power when comparing two supplement types. Calculate the required sample size per group.
#@#
#@# D) Thoroughly check the assumptions of ANOVA for your analysis in Part A. Create appropriate diagnostic plots and conduct formal tests where applicable. State whether the ANOVA assumptions are adequately met.
#@# --END OF QUESTION_3--
#@# --START OF ANSWER_3--
#@# PASTE R CODE BELOW HERE

summary(ToothGrowth)
#first lets check for missing values
sum(is.na(ToothGrowth))          
colSums(is.na(ToothGrowth))

#as there are no missing values we can take the dataset as it is

#3-A

# Combine supplement and dose into a "treatment" categorical variable using which we'll perform the ANOVA
#first Data
ToothGrowth <- ToothGrowth %>% 
  mutate(treatment = paste(supp, dose, sep="_"))

# NULL Hypothesis (H0) - Average tooth growth is same across all six treatment combinations

# Compare length over 6 treatment combinations  
growth_anova <- aov(len ~ treatment, data = ToothGrowth)
summary(growth_anova)

# p-value - <2e-16 (very small) (Reject Null Hypothesis)
# F-Statistic - 41.56 (large)(indicates a greater evidence of differences between groups )

#so we can say that Average tooth growth is different across all six treatment combinations
# Also, the LARGE F-Value indicates a greater evidence of differences between groups
# Therefore, a seperate TukeyHSD post-hoc test will be required to determine which groups differ

#3-B

# post-hoc test - TukeyHSD
tukeyResults <- TukeyHSD(growth_anova)
tukeyResults

# For each treatement group, check which specific pairs differ significantly
tukeyResults$treatment[,"p adj"] < 0.05


#3-C

# Using power analysis to find the Sample size for desired power 80% with effect size d=0.5
power_analysis_n <- pwr.t.test(d=0.5, power=0.8, sig.level=0.05, type="two.sample")
power_analysis_n
# From this power analysis, we conclude that the reqired sample size for each group is  n = 63.76 or minimum of 64.

#3-D

# Assumptions of ANOVA and diagnostic plots

#  Residual Diagnostic Plots
par(mfrow = c(2, 2))
plot(growth_anova)

# 1. Normality of residuals
shapiro.test(residuals(growth_anova))  
#as p value(0.6694) is greater than 0.05 residuals are approximately normal

# 2. Homogeneity of variance (Levene's Test)
library(car)
leveneTest(len ~ treatment, data = ToothGrowth)  
#as p value(0.6694) is greater than 0.05 , equal variance is assumed.


#The assumptions are adequately met

#@# --END OF ANSWER_3--
#@#=================================
#@# --START OF QUESTION_4--
#@# Title: Global Development Patterns (gapminder dataset)
#@# You are a development economist studying patterns in global health and economic indicators across different continents.
#@#
#@# A) Create a categorical variable that classifies countries as having "High" (above median) or "Low" (below median) life expectancy for the year 2007. Test whether there is a significant association between continent and this life expectancy classification.
#@#
#@# B) For the year 2007, test whether the distribution of countries across GDP per capita quartiles follows a uniform distribution (equal numbers in each quartile). State your null hypothesis and interpret your results.
#@#
#@# C) Calculate an appropriate effect size measure for the association you tested in Part A. Interpret what this tells you about the practical significance of the relationship between continent and life expectancy classification.
#@#
#@# D) Examine how the number of countries with "High" life expectancy has changed over time across all years in the dataset. Describe the overall trend and identify any notable patterns.
#@# --END OF QUESTION_4--
#@# --START OF ANSWER_4--
#@# PASTE R CODE BELOW HERE

#first lets check for missing data
#first lets check for missing values
sum(is.na(gapminder))          
colSums(is.na(gapminder))

#as there are no missing values we take the data as it is

#4-A

#First as always lets get the required data
gap_2007 <- gapminder %>%
  filter(year == 2007)

#median
median_lifeExp <- median(gap_2007$lifeExp, na.rm = TRUE)

#classifying
gap_2007 <- gap_2007 %>%
  mutate(lifeExp_cat = ifelse(lifeExp >= median_lifeExp, "High", "Low"))

table(gap_2007$lifeExp_cat)
#as both are 71 each we are correct

contingency_table <- table(gap_2007$continent, gap_2007$lifeExp_cat)
print(contingency_table)

#using chi-square test 
chisq.test(contingency_table)# null hypothesis is that Life expectancy category is independent of continent.

#as p-value is significantly lower than 0.05 that means
#continent is associated with life expectancy classification

#4-B

#null hypothesis would be that the number of countries in each GDP per capita quartile is equal.
#Alternate hypothesis would be that the number of countries in each GDP per capita quartile is not equal.

gap_2007 <- gap_2007 %>%
  mutate(gdp_quartile = cut(gdpPercap,
                            breaks = quantile(gdpPercap, probs = seq(0, 1, 0.25), na.rm = TRUE),
                            include.lowest = TRUE, labels = FALSE))#from today's class

# Test uniformity (expected = equal frequency)
quartile_table <- table(gap_2007$gdp_quartile)
chisq.test(quartile_table)

#as p-value is 0.998 which is greater than 0.05 we fail to reject  null hypothesis.

#4-C
#caluculating effect size
library(rcompanion)

cramerV(contingency_table)

#as cramer v is 0.6795 we can say that
#There is a strong association between continent and life expectancy category (High vs Low).

#4-D
#took little help from gpt but still got same plot
#assuming that data is fake we get this result for this data
gapminder %>%
  group_by(year) %>%
  mutate(lifeExp_cat = ifelse(lifeExp >= median(lifeExp), "High", "Low")) %>%
  count(year, lifeExp_cat) %>%
  filter(lifeExp_cat == "High") %>%
  ggplot(aes(x = year, y = n)) +
  geom_line(color = "steelblue", size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Number of High Life Expectancy Countries Over Time",
       x = "Year", y = "Count of High Life Expectancy Countries")

#There is no change over time 
#that is the number of countries classified as having high life expectancy remains constant across all years.


#@# --END OF ANSWER_4--
#@#=================================
#@# --START OF QUESTION_5--
#@# Title: Diamond Pricing Analysis (diamonds dataset)
#@# You are a data scientist for a jewelry company developing a pricing model for diamonds based on their characteristics.
#@#
#@# A) Build a simple linear regression model predicting diamond price based on carat weight. Interpret the slope coefficient in business terms and assess the model's explanatory power.
#@#
#@# B) Expand your model to include carat, cut, color, and clarity as predictors. Conduct a thorough diagnostic analysis including residual plots, normality tests, and identification of influential observations.
#@#
#@# C) Assess whether multicollinearity is a concern in your multiple regression model by calculating variance inflation factors. If multicollinearity exists, propose and implement a solution. Then use an information criterion approach to determine the optimal set of predictors.
#@#
#@# D) Using your final model, predict the price of a 1.5-carat, Premium cut, G color, VS1 clarity diamond. Provide both a point prediction and a 95% prediction interval. Interpret what this interval tells you about the uncertainty in predicting individual diamond prices.
#@# --END OF QUESTION_5--
#@# --START OF ANSWER_5--
#@# PASTE R CODE BELOW HERE

#first lets check for missing values
sum(is.na(diamonds))          
colSums(is.na(diamonds))

#as there are no missing data we can take the data as it is

#5-A

# Lets first Fit model
model_simple <- lm(price ~ carat, data = diamonds)
summary(model_simple)
coefficients(model_simple)

# visualizing for better understanding
ggplot(diamonds, aes(x = carat, y = price)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", col = "blue") +
  labs(title = "Linear Regression: Price vs Carat", x = "Carat", y = "Price")

#The positive slope makes sense: as carat weight increases, price generally increases
#I feel like model captures the general trend well, but it is kind of over predicting at the tail end.

#the slope says that for every 1 carat increase in diamond weight,
#the price increases by approximately $7756, on average, assuming all other factors (like cut, clarity, color) are held constant.

#Intercept of -2256.36 is not possible in real life
# and we can’t have a diamond with 0 carat as well but assuming it was 0 we get the intercept value.


#R square value is 0.8493 and adjusted R squared value is also 0.8493
# the value of 0.8493 means the model is explaining about 84.93% of the variance in diamond price based only on carat weight.


#5-B

#same as before we fit the model

model_full <- lm(price ~ carat + cut + color + clarity, data = diamonds)
summary(model_full)

# Plotting
par(mfrow = c(2, 2))
plot(model_full)
par(mfrow = c(1, 1))

#these are the residual plots and we can say that there is no linearity,no normality and there is outlier influence.

# Checking normality
shapiro.test(sample(residuals(model_full), 5000)) #as size must be between 3 and 5000
#as p value is very  low compared to 0.05 we reject the null hypothesis and can say that
#it is not normally distributed

# For influential observations
influencePlot(model_full, id.method = "identify")

#it labelled the most influential points
#Points 16284, 27631, 27416 have High leverage, High Cook's D and Large residuals.
#High leverage means extreme values (Leverage is for predictors)
#Cook's D gives influence or how much that point pulls the line


#5-C
#for multi-collinearity we use VIF
vif(model_full)

# as vif value is less than 5 there is no multi-collinearity
#if multi-collinearity existed we would have to use step-wise model ,there are 2 approaches-
#adding features from scratch or removing feature one-by-one from all features

#assuming multi-collinearity
library(MASS)
model_step <- stepAIC(model_full, direction = "both", trace = TRUE)
#this first takes a model with all features and then drops features one by one based on information gained from AIC.
#ower the AIC value the better the model is

summary(model_step)
#5-D
#for predicting new diamond, first lets create a df for new diamond

new_diamond <- data.frame(carat = 1.5,
                          cut = "Premium",
                          color = "G",
                          clarity = "VS1")

# To predict
predict(model_step, newdata = new_diamond, interval = "prediction")

#the outputs mean that
#fit is the predicted value which is 10864.47
#and these lwr and upr are 95% confidence interval range for that new diamond

#@# --END OF ANSWER_5--
#@#=================================