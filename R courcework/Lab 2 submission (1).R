#@#========================================
#@#
#@# LAB 2: Data Wrangling with DPLYR
#@#        Data Visualization with GGPLOT
#@#
#@#========================================
#@# --START OF QUESTION_1--
#@# Title: F1 Database Single Tables
#@#
#@# PART A: 
#@# Find all drivers whose nationality is Chinese, listing their first name and last name.
#@#
#@# PART B: 
#@# Create a new column called `full_name` that combines forename and surname with a space between them.
#@#
#@# PART C: 
#@# Count how many drivers there are from each nationality, and arrange by count, in descending order.
#@# Which nationalities have had just one driver?
#@#
#@# PART D: 
#@# Create a new column that shows driver surnames in ALL CAPS.
#@# Show all surnames starting with "S"
#@# Show all surnames ending with "S"
#@# Show all surnames containing a double "S"
#@# Use stringr for this.
#@#
#@# PART E: 
#@# Convert date to proper format using lubridate, extract month and day of week, and list all the 2020 July/August races.
#@# How many of these were on a Saturday?
#@# 
#@# Who is the youngest driver?
#@#
#@# --END OF QUESTION_1--
#@# --START OF ANSWER_1--
#@# PASTE R CODE BELOW HERE

# Load required libraries
library(tidyverse)

# Load data files
circuits <- read_csv("circuits.csv")
constructors <- read_csv("constructors.csv")
drivers <- read_csv("drivers.csv")
races <- read_csv("races.csv")
results <- read_csv("results.csv")
pit_stops <- read_csv("pit_stops.csv")


#PART A

q1_A <- drivers %>%
  filter(nationality=="Chinese")%>%
  select(nationality,forename,surname) 


q1_A

# PART B: 
q1_B <- drivers %>%
  mutate(full_name = paste(forename, surname, sep = " ")) 

q1_B
q1_B%>% select(full_name)

# PART C: 

q1_c <- drivers %>%
  group_by(nationality) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
q1_c

one_driver <- q1_c %>%
  filter(count==1)
one_driver

# PART D: 
q1_D <- drivers %>%
  mutate(surname_caps = str_to_upper(surname))
q1_D

q1_D_start_s <- q1_D %>%
  filter(str_starts(surname, "S"))
q1_D_start_s

q1_D_end_s <- q1_D %>%
  filter(str_ends(surname, "S"))
q1_D_end_s

q1_D_double_s <- q1_D %>%
  filter(str_detect(surname, "ss"))
q1_D_double_s

# PART E: 
q1_E <- races %>%
  mutate(
    date = mdy(date),
    month = month(date, label = TRUE),
    weekday = wday(date, label = TRUE)
  ) %>%
  filter(year == 2020, month %in% c("Jul", "Aug"))
q1_E

q1_E %>% select(weekday)

q1_E_saturdays <- q1_E %>%
  filter(weekday == "Sat") %>%
  nrow()
q1_E_saturdays #all races on sunday

youngest_driver <- drivers %>%
  mutate(dob = mdy(dob)) %>%
  filter(!is.na(dob)) %>%
  arrange(desc(dob)) %>%
  slice_head(n=1)
youngest_driver



#@# --END OF ANSWER_1--
#@#=================================
#@# --START OF QUESTION_2--
#@# Title: F1 Database Multi-Table
#@#
#@# PART A:
#@# For all drivers who ever made it to the podium (finished 1st, 2nd, or 3rd),
#@# list their names alongside their position, points, and the date and place of the race
#@# (for every race where they made it to the podium), starting from most recent date
#@# and going backward in time.
#@#
#@# PART B: 
#@# What are the mean and maximum values for both points and laps, across the entire set of results (ever).
#@# Use the across function to do this.
#@#
#@# PART C:
#@# For each driver, compare the points earned in each race to their points their previous race. 
#@# Show driver names, race information, current race points, previous race points, and the points difference.
#@#
#@# PART D:
#@# Find drivers who have achieved a fastest lap time in at least 3 different races.
#@# Show their names, and the number of races in which they achieved the fastest laps.
#@# Sorted this information from most podium appearances to least.
#@#
#@# PART E:
#@# For each nationality, find the driver with the most race wins.
#@# Include their name, and calculate some statistics about their racing performance, like their ave wins and points.
#@# 
#@# --END OF QUESTION_2--
#@# --START OF ANSWER_2--
#@# PASTE R CODE BELOW HERE

# Load required libraries
library(tidyverse)

# Load data files
circuits <- read_csv("circuits.csv")
constructors <- read_csv("constructors.csv")
drivers <- read_csv("drivers.csv")
races <- read_csv("races.csv")
results <- read_csv("results.csv")
pit_stops <- read_csv("pit_stops.csv")

# PART A:
joined_data1 <- results %>%
  left_join(drivers,by = "driverId")
joined_data2 <- joined_data1 %>%
  left_join(races,by = "raceId")           
joined_data2

q2_A <- joined_data2 %>%
  mutate(date = mdy(date)) %>% 
  filter(positionOrder %in% c(1, 2, 3)) %>%
  arrange(desc(date)) %>%
  select(forename, surname, position = positionOrder, points, date, name)

q2_A 

# PART B:
q2_B <- results %>%
  summarise(
    across(c(points, laps),
           list(mean = ~mean(.x, na.rm = TRUE),
                max = ~max(.x, na.rm = TRUE)))
  )

q2_B

# PART C:
results <- results %>%
  arrange(driverId,raceId) #to get from driverid 1 and raceid 1 to end in order


q2_C <- results %>%
  group_by(driverId) %>%
  arrange(raceId) %>%
  mutate(prev_points = lag(points),
         point_change = points - prev_points) %>%
  left_join(drivers, by = "driverId") %>%
  left_join(races, by = "raceId") %>%
  select(forename, surname, race_name =name, round, raceId, current_points =points, prev_points, point_change)
q2_C

# PART D:
fastest_laps_ranked <- results %>%
  filter(!is.na(fastestLapTime)) %>%
  group_by(raceId) %>%
  mutate(fastest_lap_rank = rank(fastestLapTime, ties.method = "first")) %>%
  ungroup() %>%
  filter(fastest_lap_rank == 1) %>%
  group_by(driverId) %>%
  summarise(num_fastest_laps = n_distinct(raceId), .groups = "drop") %>%
  filter(num_fastest_laps >= 3)


podium_counts <- results %>%
  filter(positionOrder <= 3) %>%
  group_by(driverId) %>%
  summarise(num_podiums = n(), .groups = "drop")


q2_D <- fastest_lap_counts %>%
  left_join(podium_counts, by = "driverId") %>%
  left_join(drivers, by = "driverId") %>%
  arrange(desc(num_podiums)) %>%
  select(forename, surname, num_fastest_laps, num_podiums)


q2_D

# PART E:

#average wins ??

per_nation_max_winner <- results %>%
  filter(positionOrder == 1) %>%
  group_by(driverId) %>%
  summarise(total_wins = n(), .groups = "drop") %>%
  left_join(drivers %>% select(driverId, forename, surname, nationality), by = "driverId") %>%
  group_by(nationality) %>%
  slice_max(total_wins, with_ties = FALSE) %>%
  ungroup()
per_nation_max_winner


performance_stats <- results %>%
  filter(driverId %in% per_nation_max_winner$driverId) %>%
  group_by(driverId) %>%
  summarise(
    total_wins=sum(positionOrder == 1, na.rm = TRUE),
    num_podiums = sum(positionOrder <= 3, na.rm = TRUE),
    total_races = n(),
    total_points = sum(points, na.rm = TRUE),
    avg_points_per_race = mean(points, na.rm = TRUE),
    avg_finishing_position = mean(positionOrder, na.rm = TRUE),
    avg_starting_position = mean(grid, na.rm = TRUE),
    avg_wins = total_wins / total_races
  ) %>%
  ungroup()
performance_stats

q2_E <- per_nation_max_winner %>%
  left_join(performance_stats, by = "driverId") %>%
  mutate(full_name = paste(forename, surname)) %>%  
  select(
    nationality,
    forename,
    surname,
    total_wins =total_wins.x,
    num_podiums,
    total_races,
    total_points,
    avg_points_per_race,
    avg_finishing_position,
    avg_starting_position,
    avg_wins
  ) %>%
  arrange(desc(total_wins))

q2_E


#@# --END OF ANSWER_2--
#@#=================================
#@# --START OF QUESTION_3--
#@# Title: F1 Data Integration
#@#
#@# PART A: 
#@# Analyze how driver nationality diversity has changed over F1 history. 
#@# For races from 1970 onwards, count the number of drivers per nationality+decade, and use this to answer:
#@# For each nationality, how many drivers did they have in each decade?
#@# For each nationality, how many decades have they been in F1?
#@#
#@# PART B: 
#@# Examine how well qualifying positions translate to race results across different circuits. 
#@# For each circuit, calculate the difference between qualifying and finishing positions of the drivers 
#@# who raced on that circuit.  Calculate average position change (between qualifying ane race position)
#@# and identify circuits where qualifying performance matters the most vs least.
#@#
#@# --END OF QUESTION_3--
#@# --START OF ANSWER_3--
#@# PASTE R CODE BELOW HERE

# Load required libraries
library(tidyverse)
library(lubridate)

# Load data files
circuits <- read_csv("circuits.csv")
constructors <- read_csv("constructors.csv")
constructor_results <- read_csv("constructor_results.csv")
constructor_standings <- read_csv("constructor_standings.csv")
drivers <- read_csv("drivers.csv")
driver_standings <- read_csv("driver_standings.csv")
races <- read_csv("races.csv")
results <- read_csv("results.csv")
pit_stops <- read_csv("pit_stops.csv")
qualifying <- read_csv("qualifying.csv")

# PART A:

race_needed <-races %>% 
  select(raceId, circuitId)


joined_data <- results %>%
  left_join(drivers, by = "driverId") %>%
  left_join(races %>% select(raceId, year), by = "raceId") %>%
  filter(year >= 1970) %>%
  mutate(decade = paste0(year - (year %% 10), "s"))  
joined_data



q3_A1 <- joined_data %>%
  group_by(nationality, decade) %>%
  summarise(num_drivers = n_distinct(driverId))%>%
  ungroup()

q3_A2 <- drivers_per_nationality_decade %>%
  group_by(nationality) %>%
  summarise(num_decades = n_distinct(decade)) %>%
  ungroup()

q3_A1
q3_A2


# PART B:

qualifying_needed <- qualifying %>%
  select(raceId, driverId, qualifying_position = position)

race_needed <-races %>% 
  select(raceId, circuitId)

circuits_needed <- circuits %>% 
  select(circuitId, circuit_name = name)


position_difference <- results %>%
  select(raceId, driverId, finishing_position = positionOrder) %>%
  left_join(qualifying_needed, by = c("raceId", "driverId")) %>%
  filter(!is.na(qualifying_position), !is.na(finishing_position)) %>%
  mutate(position_change = qualifying_position - finishing_position) %>%
  left_join(race_needed, by = "raceId") %>%
  left_join(circuits_needed, by = "circuitId")

position_difference


q3_B <- position_difference %>%
  group_by(circuit_name) %>%
  summarise(
    avg_position_change = mean(position_change, na.rm = TRUE),
  ) %>%
  ungroup() %>%
  arrange(avg_position_change)

q3_B 
q3_B_top <- position_change_by_circuit %>%
  slice_max(avg_position_change, n = 5)

q3_B_top#qualifying performance matters most here because avg_position_change is 0.
#which means qualification position = final position(on average). 
#So need to qualify in top 3 to end race on podium

q3_B_bottom <- position_change_by_circuit %>%
  slice_min(avg_position_change, n = 5)

q3_B_bottom #qualifying performance matters least here(in comparision to other tracks) because avg_position_change is not 0.
#which means qualification position != final position(on average), so overtaking is more than possible. 



#@# --END OF ANSWER_3--
#@#=================================
#@# --START OF QUESTION_4--
#@# Title: Visualization Part 1
#@#
#@# PART A: 
#@# Create a scatter plot showing the relationship between circuit altitude and latitude.
#@# Add a smooth trend line and color points by country. Include proper labels.
#@#
#@# PART B: 
#@# Create a histogram showing the distribution of driver ages. 
#@# Use 20 bins and add a vertical line showing the mean age.
#@#
#@# PART C: 
#@# Code is provided to help you know which circuits are permanent tracks versus streets.
#@# Create box plots comparing fastest lap times across different circuit types
#@# (street circuits vs permanent tracks). Add a jittered points overlay.
#@#
#@# PART D:
#@# Data is provided: Average lap times by year
#@# Create a line plot showing how average fastest lap times have changed over the years
#@# for the Monaco circuit. Add different colors for different decades.
#@#
#@# PART E: 
#@# Create a heatmap showing correlations between circuit latitude, longitude, and altitude.
#@# Use a diverging color scale and add correlation values as text.
#@#
#@# --END OF QUESTION_4--
#@# --START OF ANSWER_4--
#@# PASTE R CODE BELOW HERE

# Load required libraries
library(tidyverse)
library(patchwork)
library(plotly)

# Load F1 data
circuits <- read_csv("circuits.csv")
constructors <- read_csv("constructors.csv")
drivers <- read_csv("drivers.csv")
races <- read_csv("races.csv")
results <- read_csv("results.csv")
pit_stops <- read_csv("pit_stops.csv")
qualifying <- read_csv("qualifying.csv")

# PART A:
cleaned_circuits <- circuits %>%
  filter(!is.na(lat), !is.na(alt))

ggplot(cleaned_circuits, aes(x = lat, y = alt, color = country)) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  labs(
    title = "Circuit Altitude vs Latitude",
    x = "Latitude (°)",
    y = "Altitude (m)",
    color = "Country"
  )


# PART B:
age_drivers <- drivers %>%
  mutate(
    dob = mdy(dob),  
    age = as.numeric(difftime(Sys.Date(), dob, units = "days")) / 365.25
  )

age_drivers

mean_age <- mean(age_drivers$age, na.rm = TRUE)
mean_age

ggplot(age_drivers, aes(x = age)) +
  geom_histogram(bins = 20) +
  geom_vline(xintercept = mean_age, linetype = "dashed", color = "red", size = 1) +
  labs(
    title = " Driver Ages",
    x = "Age (years)",
    y = "Number of Drivers"
  ) 


# PART C:

# Data preparation
circuit_types <- circuits %>%
  mutate(circuit_type = case_when(
    str_detect(tolower(name), "street|monte|monaco|baku|singapore") ~ "Street Circuit",
    TRUE ~ "Permanent Track"
  )) %>%
  select(circuitId, circuit_type)
data <- 
  results %>%
  filter(!is.na(fastestLapTime)) %>%
  group_by(raceId) %>%
  mutate(fastest_lap_rank = rank(fastestLapTime, ties.method = "first")) %>%
  ungroup() %>%
  filter(fastest_lap_rank == 1)%>%
  inner_join(races , by= "raceId" ) %>%
  inner_join( circuit_types , by = "circuitId") %>%
  select(fastestLapTime,raceId,circuitId, circuit_type)

data  


q_4c <-ggplot(data, aes(x = circuit_type, y = fastestLapTime, fill = circuit_type)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  labs(
    title = "Fastest lap times across different circuit types",
    x = "Circuit Type",
    y = "Fastest Lap Time ",
    fill = "Circuit Type",
  )

q_4c


# PART D:

monaco_race_ids <- races %>%
  filter(str_detect(tolower(name), "monaco")) %>%
  pull(raceId)
monaco_race_ids

race_needed <- races %>%
  select(raceId, year)

monaco_fast_laps <- results %>%
  filter(raceId %in% monaco_race_ids, !is.na(fastestLapTime)) %>%
  left_join(race_needed, by = "raceId") %>%
  mutate(
    
    lap_sec = sapply(fastestLapTime, toSeconds),
    decade = paste0(year - (year %% 10), "s")  
  ) %>%
  group_by(year, decade) %>%
  summarise(avg_lap_time = mean(lap_sec, na.rm = TRUE), .groups = "drop")  # fixed summarise

monaco_fast_laps

ggplot(monaco_fast_laps, aes(x = year, y = avg_lap_time, color = decade, group = decade)) +
  geom_line(size = 1.3) +
  labs(
    title = "Average Fastest Lap Times at Monaco",
    x = "Year",
    y = "Average Fastest Lap",
    color = "Decade"
  )


# PART E:

library(reshape2)


cor_df <- circuits %>%
  select(lat, lng, alt) %>%
  filter(!is.na(lat), !is.na(lng), !is.na(alt))


cormat <- round(cor(cor_df), 2)


melted <- melt(cormat, na.rm = TRUE)


ggplot(melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = value), color = "black", size = 4) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red",
    midpoint = 0, limit = c(-1, 1),
    name = "Pearson\nCorrelation")  +
  labs(
    title = "Correlation Between Latitude, Longitude & Altitude",
    x = "", y = ""
  )+
  coord_fixed()

#@# --END OF ANSWER_4--
#@#=================================
#@# --START OF QUESTION_5--
#@# Title: F1 Data Visualization 
#@#
#@# PART A: 
#@# Create a series of violin plots, one per decade to show the distribution of 
#@# points achieved by the drivers who raced in each decades.
#@# Add box plots inside the violins and mean points.
#@#
#@# PART B: 
#@# Create a 2D density plot showing the relationship between qualifying and finishing positions.
#@# Use filled contours and add the diagonal line where quali = finish position.
#@#
#@# PART C: 
#@# Data is provided: Circuit characteristics analysis
#@# Create a composite plot with 3 panels:
#@# - Left: Scatter plot of circuit latitude vs longitude, colored by altitude
#@# - Top right: Histogram of circuit altitudes
#@# - Bottom right: Bar chart of circuits by country (top 8 countries)
#@# Combine using patchwork with proper layout.
#@#
#@# PART D: 
#@# Create a scatter plot showing constructor total points by year for the top 5 constructors
#@# (By top 5 constructors, I mean the constructors with the greatest number of points total, today)
#@# Add a text annotation highlighting the best season ever (highest points in a single year) 
#@# showing both the constructor name and points value using annotate().
#@# 
#@# --END OF QUESTION_5--
#@# --START OF ANSWER_5--
#@# PASTE R CODE BELOW HERE

# Load required libraries
library(tidyverse)
library(patchwork)
library(plotly)

# Load F1 data
circuits <- read_csv("circuits.csv")
constructors <- read_csv("constructors.csv")
drivers <- read_csv("drivers.csv")
races <- read_csv("races.csv")
results <- read_csv("results.csv")
pit_stops <- read_csv("pit_stops.csv")
qualifying <- read_csv("qualifying.csv")

# PART A:

race_needed <-races %>% 
  select(raceId, year)
drivers_needed <- drivers %>% 
  select(driverId, forename, surname)

q5_A <- results %>%
  left_join(race_needed, by = "raceId") %>%
  left_join(drivers_needed, by = "driverId") %>%
  mutate(decade = paste0(year - (year %% 10), "s"))

ggplot(q5_A, aes(x = decade, y = points, fill = decade)) +
  geom_violin(alpha = 0.5, trim = FALSE) +
  geom_boxplot(width = 0.1, fill = "white", outlier.size = 1) +
  stat_summary(fun = mean, geom = "point", color = "red", size = 3) +
  labs(
    title = "Distribution for Race Points per Decade",
    x = "Decade",
    y = "Driver Points per Race"
  )  +
  theme(legend.position = "none")

# PART B:

results_needed <-results %>% 
  select(raceId, driverId, finish_pos = positionOrder)

q5_B <- qualifying %>%
  select(raceId, driverId, qual_pos = position) %>%
  inner_join(results_needed, by = c("raceId", "driverId"))

ggplot(q5_B, aes(x = qual_pos, y = finish_pos)) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", contour = TRUE, alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(
    title = "2D Density plot for  Qualifying vs Finish Positions",
    x = "Qualifying Position",
    y = "Finishing Position",
    fill = "Density"
  ) +
  scale_fill_viridis_c()

# PART C:

p1_scatter <- ggplot(circuits, aes(x = lng, y = lat, color = alt)) +
  geom_point(size = 3) +
  scale_color_viridis_c() +
  labs(
    title = "Circuit Map (Altitude Colored)",
    x = "Longitude",
    y = "Latitude"
  ) 

# TOP RIGHT: Histogram of altitudes
p2_hist <- ggplot(circuits, aes(x = alt)) +
  geom_histogram(fill = "steelblue", bins = 20, color = "white") +
  labs(title = "Circuit Altitude Distribution", x = "Altitude (m)", y = "Count") 

# BOTTOM RIGHT: Bar chart of circuits per top 8 countries
top_countries <- circuits %>%
  count(country, sort = TRUE) %>%
  top_n(8, n)

p3_bar <- circuits %>%
  filter(country %in% top_countries$country) %>%
  count(country) %>%
  ggplot(aes(x = reorder(country, -n), y = n, fill = country)) +
  geom_col(show.legend = FALSE) +
  coord_flip()

# Combine plots
(p1_scatter | (p2_hist / p3_bar)) + plot_layout(widths = c(2, 1))

# PART D:
top_constructors <- constructor_standings %>%
  group_by(constructorId) %>%
  summarise(total_points = sum(points, na.rm = TRUE), .groups = "drop") %>%
  top_n(5, total_points) %>%
  pull(constructorId)

best_season <- constructor_standings %>%
  filter(constructorId %in% top_constructors) %>%
  left_join(races %>% select(raceId, year), by = "raceId") %>%
  left_join(constructors %>% select(constructorId, name), by = "constructorId") %>%
  group_by(name, year) %>%
  summarise(yearly_points = sum(points, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(yearly_points)) %>%
  slice(1)
best_season

ggplot(constructor_points, aes(x = year, y = points, color = name)) +
  geom_point(alpha = 0.7) +
  stat_summary(fun = sum, geom = "line", size = 1) +
  annotate(
    "text", x = Inf, y = Inf,
    label = paste0("Best: ", best_season$name, "\n", best_season$yearly_points, " pts"),
    hjust = 1.1, vjust = 1.1,
    size = 4, color = "black"
  )  +
  labs(
    title = "Constructor Points Over all Time (Top 5)",
    x = "Year",
    y = "Points",
    color = "Constructor"
  )


# PART E:


#@# --END OF ANSWER_5--
