patient_ids <- c(101, 102, 103, 104, 105, 106)
temperatures <- c(98.6, 99.1, 97.8)
#A
temperatures_new <- temperatures + 1.5
temperatures_new
#B
logical_vector <- temperatures_new > 100
#C
high_fever <- patient_ids[logical_vector]
high_fever
#D
#the vector with smaller size(which is temperatures_new with size 3) replicated it self to match the size of vector patient_id as they both are not the same size,.
#So temperatures vector recycles from (98.6, 99.1, 97.8) to (98.6, 99.1, 97.8,98.6, 99.1, 97.8) to match patient_id vetor length which is 6.So we get 6 results(as there are 6 patient_id's) eventhough we have 3 temperatures values.

#####################################################################################################

#A
patients<- data.frame(
  name= c("Alice", "Bob", "Carol", "David"),
  age= c(25, 67, 45, 33),
  has_insurance= c(TRUE, TRUE, FALSE, TRUE)
  
)
patients
#B 
patients$age

#C
patients[2,3]

#D
patients[, c("name", "has_insurance")]

#E
str(patients)
# A data frame is a list of vectors, when we use str()for our data frame it shows structure of each column.The`name`column is a character vector = output shows "chr"
# The `age` column is a numeric vector = output shows "num"
# The `has_insurance` column is a logical vector(Boolean) = output shows "logi"
# So, each column in the data frame is a vector, and str() reveals their types.

############################################################################

#A
library(tidyverse)
wait_times <- tibble(
  patient = c("Sarah", "Mike", "Lisa", "Tom", "Emma"),
  wait_minutes = c(15, 45, 8, 32, 12),
  department = c("Cardiology", "Emergency", "Cardiology", "Emergency", "Cardiology")
)
wait_times

#B 
wait_times %>% 
  filter(department == "Emergency") %>%
  arrange(desc(wait_minutes)) %>%
  select(patient,wait_minutes)

#C
wait_times %>% 
  group_by(department) %>%
  summarize(average_wait= mean(wait_minutes))

#D

wait_times %>% 
  mutate(wait_category = if_else(wait_minutes <20,"Short","Long"))

##################################################################################

#A

patient_names <- tibble(
          id = c(1, 2, 3),
          name = c("  JOHN DOE  ", "jane_smith", "Bob O'Connor")
)
appointments <- tibble(
          patient_id = c(1, 2, 4),
          appointment_date = c("2023-07-15", "2023-07-16", "2023-07-17")
)
patient_names
appointments
#B 

patient_names_cleaned <- patient_names %>%
  mutate(
    name = name %>%
      str_trim() %>%
      str_replace_all("_", " ") %>%
      str_to_title()
  )
patient_names_cleaned

#C
join_clean_appointment <- left_join(patient_names_cleaned, appointments, by = c("id" = "patient_id"))
join_clean_appointment

#D
# left_join joins keeps all the entries on the left side and matches them to entries on right data even though few entries on left side do not match with entries on right side.It returns NA if there is no match on right side data.the entries on right side which do not match with left side data are not considered when using left join.
# When we left joined the cleaned patient names data (left side) with the appointments data (right side),there was no match for Bob (id = 3) because his id was not present in the appointments data.
# So, the appointment_date for Bob is NA, and as patient with id=4 does not match with left side data, he is not involved in the joined data.
#############################################################################################

#A
survey_responses <- tibble(
  response_id = 1:8,
  patient_name = c("Alice Johnson", "bob smith", "Carol Davis", "alice johnson", 
                   "David Wilson", "Emma Brown", "Bob Smith", "carol davis"),
  satisfaction_score = c(5, 3, 4, 5, 2, 4, 3, 4),
  department = c("Cardiology", "Emergency", "Cardiology", "cardiology", 
                 "Emergency", "Cardiology", "emergency", "Cardiology")
)
survey_responses

#B 
survey_names_cleaned <- survey_responses %>%
  mutate(
    patient_name = patient_name %>%
      str_trim() %>%
      str_to_title(),
    department = department %>%
      str_to_title()
  )  %>% 
  select(-response_id) %>%
  distinct()
survey_names_cleaned

#C 
#1
average_satisfaction_score <- survey_names_cleaned %>%
  group_by(department) %>%
  summarize(average_satisfaction_score = mean(satisfaction_score))

average_satisfaction_score

#2
count_responses_department <- survey_names_cleaned %>%
  count(department)
count_responses_department

#3

unsatisfied_patients <- survey_names_cleaned %>%
  filter(satisfaction_score < 3)
unsatisfied_patients

#D 
#1
clinic_summary <- list(
  total_responses = nrow(survey_names_cleaned),
  departments = unique(survey_names_cleaned$department),
  avg_score = mean(survey_names_cleaned$satisfaction_score)
)
clinic_summary
#2
clinic_summary$avg_score

#E
#1
#visually we can say it removed 3 rows but in a large dataset if we have to use a logic then it would be difference in rows before and after cleaned
n_removed <- nrow(survey_responses) - nrow(survey_names_cleaned)
n_removed

#2
best_dept <- average_satisfaction_score %>%
  filter(average_satisfaction_score == max(average_satisfaction_score))%>%
  pull(department)
best_dept

#3 

#The data quality issues we encountered in this case study are inconsistent naming in patient names and department, has duplicate survey responses.



