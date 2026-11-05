# Load required libraries
library(tidyverse)
library(data.table)

# Set seed for reproducibility
set.seed(42)

# Parameters
num_subjects <- 1200
days_per_subject <- sample(5:14, num_subjects, replace = TRUE)
points_per_day <- 288 # 5-minute intervals in 24 hours

# Pre-generate subject-level clinical baseline (A1c, FPG, 2-hr OGTT)
a1c_base <- rnorm(num_subjects, mean = 5.7, sd = 0.8)
a1c_base <- pmin(pmax(a1c_base, 4.5), 10.5)

fpg_base <- a1c_base * 18.0 + rnorm(num_subjects, mean = 0, sd = 10)
ogtt_base <- a1c_base * 25.0 + rnorm(num_subjects, mean = 0, sd = 15)

# Base start timestamp (POSIXct in UTC: 2026-11-01 00:00:00)
base_start_time <- as.POSIXct("2026-11-01 00:00:00", tz = "UTC")

# Initialize storage lists
all_subject_ids <- vector("list", num_subjects)
all_timestamps <- vector("list", num_subjects)
all_raw_glucose <- vector("list", num_subjects)
all_sin_time <- vector("list", num_subjects)
all_cos_time <- vector("list", num_subjects)
all_raw_accel <- vector("list", num_subjects)
all_raw_speed <- vector("list", num_subjects)
all_a1c <- vector("list", num_subjects)
all_fpg <- vector("list", num_subjects)
all_ogtt <- vector("list", num_subjects)

cat("Generating simulation data for", num_subjects, "subjects...\n")

for (subj_idx in 1:num_subjects) {
  subj_id <- sprintf("SUBJ_%04d", subj_idx)
  n_days <- days_per_subject[subj_idx]
  n_points <- n_days * points_per_day
  
  # Stagger subject start times slightly (0 to 12 hours offset)
  start_offset <- runif(1, 0, 12 * 3600)
  t_seconds <- as.numeric(base_start_time) + start_offset + (0:(n_points - 1)) * 300
  timestamps <- as.POSIXct(t_seconds, origin = "1970-01-01", tz = "UTC")
  
  # Diurnal time features (sine/cosine over 24-hour cycle)
  sec_in_day <- t_seconds %% 86400
  sin_time <- sin(2 * pi * sec_in_day / 86400)
  cos_time <- cos(2 * pi * sec_in_day / 86400)
  
  # Simulated Raw Glucose dynamics (AR(1) process with meal spikes)
  mean_g <- fpg_base[subj_idx]
  
  # Meal peaks roughly at 8:00 (28800s), 12:30 (45000s), 18:30 (66600s)
  meal_effect <- (
    30.0 * exp(-((sec_in_day - 28800) / 3600)^2) +
      40.0 * exp(-((sec_in_day - 45000) / 3600)^2) +
      45.0 * exp(-((sec_in_day - 66600) / 3600)^2)
  )
  
  # Autoregressive noise for smooth continuous fluctuations
  noise <- numeric(n_points)
  phi <- 0.95
  for (k in 2:n_points) {
    noise[k] <- phi * noise[k-1] + rnorm(1, mean = 0, sd = 3)
  }
  
  raw_glucose <- mean_g + meal_effect + noise
  raw_glucose <- pmin(pmax(raw_glucose, 40), 400) # Physiological clamping
  
  # Simulated physical activity (sparse bursts during daytime: 7 AM to 10 PM)
  daytime_mask <- (sec_in_day > 25200) & (sec_in_day < 79200)
  accel_raw <- pmax(0, rexp(n_points, rate = 10) * daytime_mask)
  speed_raw <- pmax(0, rexp(n_points, rate = 2) * daytime_mask)
  
  # Store in lists
  all_subject_ids[[subj_idx]] <- rep(subj_id, n_points)
  all_timestamps[[subj_idx]] <- timestamps
  all_raw_glucose[[subj_idx]] <- round(raw_glucose, 2)
  all_sin_time[[subj_idx]] <- round(sin_time, 5)
  all_cos_time[[subj_idx]] <- round(cos_time, 5)
  all_raw_accel[[subj_idx]] <- round(accel_raw, 4)
  all_raw_speed[[subj_idx]] <- round(speed_raw, 4)
  all_a1c[[subj_idx]] <- rep(round(a1c_base[subj_idx], 2), n_points)
  all_fpg[[subj_idx]] <- rep(round(fpg_base[subj_idx], 1), n_points)
  all_ogtt[[subj_idx]] <- rep(round(ogtt_base[subj_idx], 1), n_points)
}


# Combine into a single data.table / tibble
df <- tibble(
  subject_id = unlist(all_subject_ids),
  timestamp_posixct = unlist(all_timestamps),
  raw_glucose = unlist(all_raw_glucose),
  hour_sin = unlist(all_sin_time),
  hour_cos = unlist(all_cos_time),
  raw_accel = unlist(all_raw_accel),
  raw_speed = unlist(all_raw_speed),
  a1c = unlist(all_a1c),
  fpg = unlist(all_fpg),
  ogtt_2hr = unlist(all_ogtt)
)

unique_id <- unique(df$subject_id)
# Apply global Z-score normalization across all rows/subjects
df1 <- df %>%
  filter(subject_id %in% sample(unique_id, 50)) |> ## get 50 to train overfit
  mutate(
    timestamp = as.POSIXct(timestamp_posixct, origin = "1970-01-01", tz = "UTC"),
    norm_gl = (raw_glucose - mean(raw_glucose)) / sd(raw_glucose),
    norm_acc = (raw_accel - mean(raw_accel)) / sd(raw_accel),
    norm_speed = (raw_speed - mean(raw_speed)) / sd(raw_speed)
  ) %>%
  select(
    subject_id,
    timestamp,
    raw_glucose,
    norm_gl,
    hour_sin,
    hour_cos,
    norm_acc,
    norm_speed,
    a1c,
    fpg,
    ogtt_2hr
  )

# Export to CSV
csv_filename <- paste0("Downloads/transformer/pseudo_cgm_data_",
                       length(unique(df1$subject_id)), "_subjects_r.csv")
fwrite(df1, csv_filename)

cat("Dataset generated and saved successfully to:", csv_filename, "\n")
cat("Total rows:", nrow(df), "\n")