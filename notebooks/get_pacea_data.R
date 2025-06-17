if(FALSE){
  # only run these lines if you want to update the package
  library(remotes)
  install.packages("gh")
  remotes::install_github("r-lib/usethis", force=TRUE)
  remotes::install_github("pbs-assess/pacea")
1}

library(tidyverse)
library(pacea)
library(lubridate)

# check date of current install
pacea_installed()

# check whether there have been any more recent updates:
# https://github.com/pbs-assess/pacea/blob/main/NEWS.md
# https://github.com/pbs-assess/pacea/network


# open help page for the package to check which variables are included
# using this to populate the settings list
help(package = "pacea")

pacea.annual <- full_join(alpi %>% dplyr::rename(Pacea_ALPI_Anomaly = anomaly),
                          npi_annual %>% dplyr::rename(Pacea_NPI_Value = value,Pacea_NPI_Anomaly = anomaly),
                          by=c("year")) %>% dplyr::rename(Year = year)
pacea.annual <- pacea.annual %>%
  filter(Year %in% 1947:2025)



# Compute winter means of NPI (Nov - Mar)
npi_winter_means <- npi_monthly %>%
  # Filter years of interest (include one extra year before the range to catch Nov of previous year)
  filter(year %in% (1947:2025)) %>%
  
  # Create a date column (use day = 1 as placeholder)
  mutate(date = make_date(year, month, 1)) %>%
  
  # Keep only November to March
  filter(month %in% c(11, 12, 1, 2, 3)) %>%
  
  # Assign "season year": e.g., Nov 1959 – Mar 1960 => 1960
  mutate(season_year = if_else(month %in% c(11, 12), year + 1, year)) %>%
  
  # Keep only seasons that end in the desired years
  filter(season_year %in% 1947:2025) %>%
  
  # Group by season and calculate mean
  group_by(season_year) %>%
  summarise(mean_value = mean(value, na.rm = TRUE)) %>%
  ungroup()

# Merge with pacea.annual
pacea.annual <- pacea.annual %>%
  left_join(npi_winter_means, by = c("Year" = "season_year")) %>%
  rename(npi_mean_NovMar = mean_value)



# Compute winter means of ONI (Dec - Feb)
oni_seasonal_means <- oni %>%
  # Include one extra year before to catch December of previous year
  filter(year %in% 1947:2025) %>%
  
  # Create a date column
  mutate(date = make_date(year, month, 1)) %>%
  
  # Keep only Dec, Jan, Feb
  filter(month %in% c(12, 1, 2)) %>%
  
  # Assign season to February year
  mutate(season_year = if_else(month == 12, year + 1, year)) %>%
  
  # Keep only seasons that end in desired years
  filter(season_year %in% 1947:2025) %>%
  
  # Group and summarise
  group_by(season_year) %>%
  summarise(mean_value = mean(value, na.rm = TRUE)) %>%
  ungroup()

# Merge with pacea.annual
pacea.annual <- pacea.annual %>%
  left_join(oni_seasonal_means, by = c("Year" = "season_year")) %>%
  rename(oni_mean_DecFeb = mean_value)


# Compute summer means of ONI (Apr - Sep)
mei_seasonal_means <- mei %>%
  # Keep relevant months and years
  filter(year %in% 1947:2025, month %in% 4:9) %>%
  
  # Create date column (optional, for consistency)
  mutate(date = make_date(year, month, 1)) %>%
  
  # Group by the same year (season is within a single calendar year)
  group_by(season_year = year) %>%
  summarise(mean_value = mean(anomaly, na.rm = TRUE)) %>%
  ungroup()

# Merge with pacea.annual
pacea.annual <- pacea.annual %>%
  left_join(mei_seasonal_means, by = c("Year" = "season_year")) %>%
  rename(mei_mean_AprSep = mean_value)


# Compute winter means of NPGO (Dec - Feb)
npgo_seasonal_means <- npgo %>%
  # Include Dec of the previous year
  filter(year %in% 1947:2025, month %in% c(12, 1, 2)) %>%
  
  # Create a date column (optional)
  mutate(date = make_date(year, month, 1)) %>%
  
  # Assign season to February's year
  mutate(season_year = if_else(month == 12, year + 1, year)) %>%
  
  # Keep only target years
  filter(season_year %in% 1947:2025) %>%
  
  # Group and summarise
  group_by(season_year) %>%
  summarise(mean_value = mean(anomaly, na.rm = TRUE)) %>%
  ungroup()

# Merge with pacea.annual
pacea.annual <- pacea.annual %>%
  left_join(npgo_seasonal_means, by = c("Year" = "season_year")) %>%
  rename(npgo_mean_DecFeb = mean_value)


# Compute winter means of NPGO (Dec - Mar)
ao_seasonal_means <- ao %>%
  # Include Dec of the previous year
  filter(year %in% 1947:2025, month %in% c(12, 1, 2, 3)) %>%
  
  # Create date (optional)
  mutate(date = make_date(year, month, 1)) %>%
  
  # Assign season to March year
  mutate(season_year = if_else(month == 12, year + 1, year)) %>%
  
  # Filter to desired output range
  filter(season_year %in% 1947:2025) %>%
  
  # Group and calculate mean
  group_by(season_year) %>%
  summarise(mean_value = mean(anomaly, na.rm = TRUE)) %>%
  ungroup()

pacea.annual <- pacea.annual %>%
  left_join(ao_seasonal_means, by = c("Year" = "season_year")) %>%
  rename(ao_mean_DecMar = mean_value)

# Compute winter and summer means of PDO (Dec - Mar and May to Sep)
pdo_winter_means <- pdo %>%
  filter(year %in% 1948:2025, month %in% c(12, 1, 2, 3)) %>%
  mutate(date = make_date(year, month, 1),
         season_year = if_else(month == 12, year + 1, year)) %>%
  filter(season_year %in% 1948:2025) %>%
  group_by(season_year) %>%
  summarise(mean_value = mean(anomaly, na.rm = TRUE)) %>%
  ungroup()

pdo_summer_means <- pdo %>%
  filter(year %in% 1948:2025, month %in% 5:9) %>%
  mutate(date = make_date(year, month, 1),
         season_year = year) %>%
  group_by(season_year) %>%
  summarise(mean_value = mean(anomaly, na.rm = TRUE)) %>%
  ungroup()

pacea.annual <- pacea.annual %>%
  left_join(pdo_winter_means, by = c("Year" = "season_year")) %>%
  rename(pdo_mean_DecMar = mean_value)

pacea.annual <- pacea.annual %>%
  left_join(pdo_summer_means, by = c("Year" = "season_year")) %>%
  rename(pdo_mean_MaySep = mean_value)

pacea.annual
write_csv(pacea.annual, "C:/Users/julia/Documents/AnglersProject/SalmonRun/SalmonRun_Project/data/GENERATED_pacea_series_annual.csv")

# currently including the following monthly variables
#	AO - Arctic Oscillation
# ENSO MEI – Multivariate ENSO Index
# NPGO – North Pacific Gyre Oscillation
# NPI – North Pacific Index monthly values
# ONI – Oceanographic Niño Index
# PDO – Pacific Decadal Oscillation


'
help(pdo)


# npi_monthly doesn not have anomaly column
# ?npi_annual states that anomalies are calculated relative to 1925-1989 mean
base.mean.monthly <- mean( npi_monthly %>% dplyr::filter(year %in% 1960:1989) %>% select(value) %>% unlist(),na.rm=TRUE)
base.mean.monthly

# different from the 1008.9 hPa value in the npi_annual help file?
# mean of the annual values matches the help file
# explanation for discrepancy? _> started issue at https://github.com/SOLV-Code/Open-Source-Env-Cov-PacSalmon/issues/118
base.mean.annual <- mean(npi_annual %>% dplyr::filter(year %in% 1960:1989) %>% select(value) %>% unlist(),na.rm=TRUE)
base.mean.annual

# also: should not it look at difference from mean for that month?
base.mean.by.month <- npi_monthly %>% dplyr::filter(year %in% 1925:1989) %>% dplyr::filter(year %in% 1925:1989) %>%
  group_by(month) %>% summarize(BaseMeanByMonth = mean(value,na.rm=TRUE))
base.mean.by.month

npi_monthly_mod <- npi_monthly %>% mutate(BaseMeanMonthly = base.mean.monthly,
                                          BaseMeanAnnual = base.mean.annual) %>%
  left_join(base.mean.by.month, by="month") %>%
  mutate(anomaly_month = value - BaseMeanMonthly,
         anomaly = value - BaseMeanAnnual, # using this as the default in the merge, for now
         anomaly_bymonth = value - BaseMeanByMonth)

npi_monthly_mod <- npi_monthly_mod  %>% arrange(year,month) %>% mutate(plot_index = c(1:dim(npi_monthly_mod)[1]))
npi_monthly_mod
write_csv(npi_monthly_mod,"DATA/DFO_PACEA_Package/GENERATED_pacea_NPI_MonthlyAnomaliesVariations.csv")


oni$value+oni$anomaly
plot(oni$value+oni$anomaly,type="l")


pacea.monthly <- full_join(ao %>% dplyr::rename(Pacea_AO_anomaly=anomaly),
                           mei %>% dplyr::rename(Pacea_MEI_anomaly=anomaly), by=c("year","month")) %>%
  full_join(npgo %>% dplyr::rename(Pacea_NPGO_anomaly=anomaly), by=c("year","month")) %>%
  full_join(npi_monthly_mod %>% select(year, month, value, anomaly) %>%
              dplyr::rename(Pacea_NPIm_value=value,
                            Pacea_NPIm_anomaly=anomaly),by=c("year","month")) %>%
  full_join(oni %>% dplyr::rename(Pacea_ONI_value=value,Pacea_ONI_anomaly=anomaly),by=c("year","month")) %>%
  full_join(pdo %>% dplyr::rename(Pacea_PDO_anomaly=anomaly),by=c("year","month")) %>%
  dplyr::rename(Year = year,Month = month)

pacea.monthly  <- pacea.monthly %>% arrange(Year,Month) %>% mutate(plot_index = c(1:dim(pacea.monthly)[1]))
pacea.monthly

write_csv(pacea.monthly,"DATA/DFO_PACEA_Package/GENERATED_pacea_series_monthly.csv")
'
