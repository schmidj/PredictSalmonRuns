library(tidyhydat)     # For Fraser tributaries
library(dataRetrieval) # For US rivers
library(dplyr)
library(lubridate)
library(readr)
library(purrr)

if(FALSE){
  # only run this to download HYDAT 
  download_hydat()
  1}

### ------ SETTINGS ------ ###
start_year <- 1947
end_year   <- 2025
canadian_rivers <- c("Chilko", "Late Stuart", "Quesnel", "Raft", "Stellako")
us_rivers       <- c("Alagnak", "Egegik", "Igushik", "Kvichak", "Naknek", "Nushagak", "Ugashik", "Wood", "Bonneville Lock & Dam")

# Helper: define month windows
summer_months <- 6:8  # June-August
spring_summer <- 4:9  # April-September

# ---- 1. FRASER RIVERS (Canada) ---- #
fraser_sites <- hy_stations(prov_terr_state_loc = "BC") %>%
  filter(grepl(paste(canadian_rivers, collapse = "|"), STATION_NAME, ignore.case = TRUE))


# Temperature & discharge (if available)
fraser_data <- map_df(fraser_sites$STATION_NUMBER, function(site) {
  message("Pulling: ", site)
  tryCatch({
    temp_df <- hy_daily_flows(site) %>%
      mutate(date = ymd(Date)) %>%
      filter(year(date) >= start_year, year(date) <= end_year) %>%
      mutate(Year = year(date), Month = month(date)) %>%
      filter(Month %in% spring_summer) %>%
      group_by(Year) %>%
      summarise(max_discharge_AprSep = max(Value, na.rm = TRUE)) %>%
      mutate(System = "Fraser", River = fraser_sites$STATION_NAME[fraser_sites$STATION_NUMBER == site],
             DataSource = "WSC")
  }, error = function(e) NULL)
})

# ---- 2. USGS RIVERS (USA) ---- #
us_sites <- whatNWISsites(stateCd = "AK", parameterCd = "00010") %>%
  filter(grepl(paste(us_rivers, collapse = "|"), station_nm, ignore.case = TRUE))

us_data <- map_df(us_sites$site_no, function(site) {
  message("USGS site: ", site)
  tryCatch({
    temp <- readNWISdv(site, parameterCd = "00010", startDate = paste0(start_year, "-01-01"),
                       endDate = paste0(end_year, "-12-31")) %>%
      rename(date = Date, temp = X_00010_00003) %>%
      mutate(Year = year(date), Month = month(date)) %>%
      filter(Month %in% summer_months) %>%
      group_by(Year) %>%
      summarise(mean_temp_JunAug = mean(temp, na.rm = TRUE)) %>%
      mutate(System = ifelse(grepl("bonneville", site, ignore.case = TRUE), "Columbia", "Bristol Bay"),
             River = us_sites$station_nm[us_sites$site_no == site],
             DataSource = "USGS")
  }, error = function(e) NULL)
})

# ---- 3. Combine and enhance ---- #

# Clean river names
river_name_map <- tibble(
  River_original = unique(c(us_sites$station_nm, fraser_sites$STATION_NAME)),
  River = c(
    "Utukok",     # 1: unrelated
    "Utukok",     # 2: unrelated
    "Utukok",     # 3: unrelated
    "Utukok",     # 4: unrelated
    "Wood",       # 5
    "Nushagak",   # 6
    "Nushagak",   # 7
    "Naknek",     # 8 (Frank Hill station)
    "Alagnak",    # 9
    "Alagnak",    # 10
    "Kvichak",    # 11
    "Alagnak",    # 12
    "Alagnak",    # 13
    "Naknek",     # 14
    "Alagnak",    # 15
    "Urban runoff", # 16: unrelated
    "Urban runoff", # 17: unrelated
    "Urban runoff", # 18: unrelated
    "Urban runoff", # 19: unrelated
    "Urban runoff", # 20: unrelated
    "Urban runoff", # 21: unrelated
    "Urban runoff", # 22: unrelated
    "Urban runoff", # 23: unrelated
    "Urban runoff", # 24: unrelated
    "Urban runoff", # 25: unrelated
    "Urban runoff", # 26: unrelated
    "Urban runoff", # 27: unrelated
    "Urban runoff", # 28: unrelated
    "Urban runoff", # 29: unrelated
    "Urban runoff", # 30: unrelated
    "Urban runoff", # 31: unrelated
    "Wood",        # 32
    "Urban runoff", # 33: person name, unrelated
    "Wood",        # 34: Woodchopper Cr, assign to Wood
    "Stellako",    # 35
    "Stellako",    # 36
    "Quesnel",     # 37
    "Quesnel",     # 38
    "Quesnel",     # 39
    "Quesnel",     # 40
    "Quesnel",     # 41
    "Quesnel",     # 42
    "Quesnel",     # 43
    "Quesnel",     # 44
    "Quesnel",     # 45
    "Raft",        # 46
    "Raft",        # 47
    "Chilko",      # 48
    "Chilko"       # 49
  )
)

# Merge with name mapping
combined_df <- bind_rows(fraser_data, us_data) %>%
  rename(River_original = River) %>%
  left_join(river_name_map, by = "River_original")

combined_df <- combined_df %>%
  relocate(System, River, River_original, Year) %>%
  arrange(System, River, Year)

# Rename system
combined_df <- combined_df %>%
  mutate(System = ifelse(System == "Fraser", "Fraser River", System))

# ---- 4. More Features from USGS (temperature and discharge) ---- #
us_extra <- map_df(us_sites$site_no, function(site) {
  message("Extra features for site: ", site)
  tryCatch({
    daily <- readNWISdv(site, parameterCd = c("00010", "00060"),
                        startDate = paste0(start_year, "-01-01"),
                        endDate = paste0(end_year, "-12-31")) %>%
      rename(date = Date) %>%
      mutate(Year = year(date),
             Month = month(date)) %>%
      group_by(Year) %>%
      summarise(
        mean_temp_MaySep = mean(X_00010_00003[Month %in% 5:9], na.rm = TRUE),
        max_temp_JunAug = max(X_00010_00003[Month %in% 6:8], na.rm = TRUE),
        max_discharge_AprSep = max(X_00060_00003[Month %in% 4:9], na.rm = TRUE),
        mean_discharge_MarMay = mean(X_00060_00003[Month %in% 3:5], na.rm = TRUE),
        River_original = us_sites$station_nm[us_sites$site_no == site],
        System = ifelse(grepl("bonneville", site, ignore.case = TRUE), "Columbia", "Bristol Bay"),
        DataSource = "USGS"
      )
  }, error = function(e) NULL)
})

# Merge new features into combined
combined_df <- combined_df %>%
  full_join(us_extra, by = c("River_original", "Year", "System", "DataSource")) %>%
  mutate(
    max_discharge_AprSep = coalesce(max_discharge_AprSep.x, max_discharge_AprSep.y)
  ) %>%
  select(-max_discharge_AprSep.x, -max_discharge_AprSep.y)

# Convert unit of discharge
combined_df <- combined_df %>%
  mutate(
    max_discharge_AprSep = if_else(
      DataSource == "USGS",
      max_discharge_AprSep * 0.0283168,  # convert cfs to m³/s
      max_discharge_AprSep               # leave HYDAT data unchanged
    )
  )

# Rearrange and export
combined_df <- combined_df %>%
  relocate(System, River, River_original, Year) %>%
  arrange(System, River, Year)

write_csv(combined_df, "C:/Users/julia/Documents/AnglersProject/SalmonRun/SalmonRun_Project/data/GENERATED_river_specific_features.csv")
