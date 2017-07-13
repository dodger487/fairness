# Chris Riederer
# July 6, 2017

# Analyze "fair" location based advertising

library(tidyverse)
library(forcats)


###############################################################################
## Load data

ny <- read_csv("nyc_all_results.csv")
ny_fash <- ny %>% filter(tag == "fashion")
# ny_fash <- read_csv("nyc_fashion_results.csv")
df <- ny_fash

la <- read_csv("la_all_results.csv")

###############################################################################
## Initial manipulations

# Whoops, add granularity
df <- df %>% mutate(idx = row_number() - 1,
                    gran_group = idx %/% (902*6),  # 902 ppl, 6 values of k
                    gran = gran_group * 0.5)

# Reorder k, set to factors
k_levels <- c(0, 0.5, 1, 2, 10, "unfair")
k_levels <- c("0", "0.5", "1", "2", "10", "unfair")
df <- df %>% mutate(k = parse_factor(df$k, k_levels))
ny <- ny %>% mutate(k = parse_factor(ny$k, k_levels))
la <- la %>% mutate(k = parse_factor(la$k, k_levels))


###############################################################################
## Data subsets

ny_health <- ny %>% filter(tag == "health")
ny_travel <- ny %>% filter(tag == "travel")
ny_fash <- df

la_fash   <- la %>% filter(tag == "fashion")
la_health <- la %>% filter(tag == "health")
la_travel <- la %>% filter(tag == "travel")


###############################################################################
## NYC Plots

# k to revenue, one line per granularity
df %>% 
  group_by(gran, k) %>% 
  summarize(revenue = sum(2*a1*result + 1*(1-a1))) %>% 
  ungroup() %>%
  mutate(gran = as.factor(gran)) %>%
  ggplot(aes(x=k, y=revenue, color=gran, group=gran)) +
  geom_point() + 
  geom_line() +
  ggtitle("NY Fashion")
ggsave("fig/nycfash_k_rev_by_gran.png")

ny_travel %>% 
  group_by(gran, k) %>% 
  summarize(revenue = sum(2*a1*result + 1*(1-a1))) %>% 
  ungroup() %>%
  mutate(gran = as.factor(gran)) %>%
  ggplot(aes(x=k, y=revenue, color=gran, group=gran)) +
  geom_point() + 
  geom_line() +
  ggtitle("NY Travel")
ggsave("fig/nyctravel_k_rev_by_gran.png")

ny_health %>% 
  group_by(gran, k) %>% 
  summarize(revenue = sum(2*a1*result + 1*(1-a1))) %>% 
  ungroup() %>%
  mutate(gran = as.factor(gran)) %>%
  ggplot(aes(x=k, y=revenue, color=gran, group=gran)) +
  geom_point() + 
  geom_line() +
  ggtitle("NY Health")
ggsave("fig/nychealth_k_rev_by_gran.png")


# k to revenue, faceted by granularity
df %>% 
  group_by(gran, k) %>% 
  summarize(revenue = sum(2*a1*result + 1*(1-a1))) %>% 
  ungroup() %>%
  mutate(gran = paste("gran = ", gran)) %>%
  ggplot(aes(x=k, y=revenue, group=gran)) +
  geom_point() + 
  geom_line() +
  facet_wrap(~ gran) +
  ggtitle("NY Health")
ggsave("fig/nycfash_k_rev_facet_gran.png")

ny_travel %>% 
  group_by(gran, k) %>% 
  summarize(revenue = sum(2*a1*result + 1*(1-a1))) %>% 
  ungroup() %>%
  mutate(gran = paste("gran = ", gran)) %>%
  ggplot(aes(x=k, y=revenue, group=gran)) +
  geom_point() + 
  geom_line() +
  facet_wrap(~ gran) +
  ggtitle("NY Travel")
ggsave("fig/nyctravel_k_rev_facet_gran.png")

ny_health %>% 
  group_by(gran, k) %>% 
  summarize(revenue = sum(2*a1*result + 1*(1-a1))) %>% 
  ungroup() %>%
  mutate(gran = paste("gran = ", gran)) %>%
  ggplot(aes(x=k, y=revenue, group=gran)) +
  geom_point() + 
  geom_line() +
  facet_wrap(~ gran) +
  ggtitle("NY Health")
ggsave("fig/nychealth_k_rev_facet_gran.png")


# Difference in a1 across groups by gran
df %>% 
  mutate(gran = as.factor(gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1)) %>% 
  summarize(a1_diff = abs(first(avg_a1) - last(avg_a1))) %>%
  ggplot(aes(x=k, y=a1_diff, color=gran, group=gran)) +
  geom_point() + 
  geom_line() +
  scale_y_continuous("Abs diff prob targeted btwn races") +
  ggtitle("NY Fashion")
ggsave("fig/nycfash_k_a1diff_by_gran.png")

ny_travel %>% 
  mutate(gran = as.factor(gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1)) %>% 
  summarize(a1_diff = abs(first(avg_a1) - last(avg_a1))) %>%
  ggplot(aes(x=k, y=a1_diff, color=gran, group=gran)) +
  geom_point() + 
  geom_line() +
  scale_y_continuous("Abs diff prob targeted btwn races") +
  ggtitle("NY Travel")
ggsave("fig/nyctravel_k_a1diff_by_gran.png")

ny_health %>% 
  mutate(gran = as.factor(gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1)) %>% 
  summarize(a1_diff = abs(first(avg_a1) - last(avg_a1))) %>%
  ggplot(aes(x=k, y=a1_diff, color=gran, group=gran)) +
  geom_point() + 
  geom_line() +
  scale_y_continuous("Abs diff prob targeted btwn races") +
  ggtitle("NY Health")
ggsave("fig/nychealth_k_a1diff_by_gran.png")


# Difference in a1 by groups
df %>% 
  mutate(gran = paste("gran = ", gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1)) %>% 
  summarize(a1_diff = abs(first(avg_a1) - last(avg_a1))) %>%
  ggplot(aes(x=k, y=a1_diff, group=gran)) +
  geom_point() + 
  geom_line() +
  facet_wrap(~gran) +
  scale_y_continuous("| Pr(targeted | white) - Pr(targeted | minority) |") +
  ggtitle("NY Fashion")
ggsave("fig/nycfash_k_a1diff_facet_gran.png")

ny_travel %>% 
  mutate(gran = paste("gran = ", gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1)) %>% 
  summarize(a1_diff = abs(first(avg_a1) - last(avg_a1))) %>%
  ggplot(aes(x=k, y=a1_diff, group=gran)) +
  geom_point() + 
  geom_line() +
  facet_wrap(~gran) +
  scale_y_continuous("| Pr(targeted | white) - Pr(targeted | minority) |") +
  ggtitle("NY Travel")
ggsave("fig/nyctravel_k_a1diff_facet_gran.png")

ny_health %>% 
  mutate(gran = paste("gran = ", gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1)) %>% 
  summarize(a1_diff = abs(first(avg_a1) - last(avg_a1))) %>%
  ggplot(aes(x=k, y=a1_diff, group=gran)) +
  geom_point() + 
  geom_line() +
  facet_wrap(~gran) +
  scale_y_continuous("| Pr(targeted | white) - Pr(targeted | minority) |") +
  ggtitle("NY Health")
ggsave("fig/nychealth_k_a1diff_facet_gran.png")


# Average a1 by race faceted by granularity
df %>% 
  mutate(gran = paste("gran = ", gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1),
            std_err = sd(a1) / sqrt(n())) %>% 
  ggplot(aes(x=k, y=avg_a1, color=race, group=race)) +
  geom_point() + 
  geom_line() +
  geom_errorbar(aes(ymin=avg_a1 - std_err, ymax=avg_a1 + std_err),
                width=0) +
  facet_wrap(~gran) +
  scale_y_continuous("Average prob. targeted") +
  ggtitle("NY Fashion")
ggsave("fig/nycfash_k_a1_facet_gran_by_race.png")

ny_travel %>% 
  mutate(gran = paste("gran = ", gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1),
            std_err = sd(a1) / sqrt(n())) %>% 
  ggplot(aes(x=k, y=avg_a1, color=race, group=race)) +
  geom_point() + 
  geom_line() +
  geom_errorbar(aes(ymin=avg_a1 - std_err, ymax=avg_a1 + std_err),
                width=0) +
  facet_wrap(~gran) +
  scale_y_continuous("Average prob. targeted") +
  ggtitle("NY Travel")
ggsave("fig/nyctravel_k_a1_facet_gran_by_race.png")

ny_health %>% 
  mutate(gran = paste("gran = ", gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1),
            std_err = sd(a1) / sqrt(n())) %>% 
  ggplot(aes(x=k, y=avg_a1, color=race, group=race)) +
  geom_point() + 
  geom_line() +
  geom_errorbar(aes(ymin=avg_a1 - std_err, ymax=avg_a1 + std_err),
                width=0) +
  facet_wrap(~gran) +
  scale_y_continuous("Average prob. targeted") +
  ggtitle("NY Health")
ggsave("fig/nychealth_k_a1_facet_gran_by_race.png")


###############################################################################
## LA Plots

# k to revenue, one line per granularity
la_fash %>% 
  group_by(gran, k) %>% 
  summarize(revenue = sum(2*a1*result + 1*(1-a1))) %>% 
  ungroup() %>%
  mutate(gran = as.factor(gran)) %>%
  ggplot(aes(x=k, y=revenue, color=gran, group=gran)) +
  geom_point() + 
  geom_line() +
  ggtitle("LA Fashion")
ggsave("fig/la_fash_k_rev_by_gran.png")

la_travel %>% 
  group_by(gran, k) %>% 
  summarize(revenue = sum(2*a1*result + 1*(1-a1))) %>% 
  ungroup() %>%
  mutate(gran = as.factor(gran)) %>%
  ggplot(aes(x=k, y=revenue, color=gran, group=gran)) +
  geom_point() + 
  geom_line() +
  ggtitle("LA Travel")
ggsave("fig/la_travel_k_rev_by_gran.png")

la_health %>% 
  group_by(gran, k) %>% 
  summarize(revenue = sum(2*a1*result + 1*(1-a1))) %>% 
  ungroup() %>%
  mutate(gran = as.factor(gran)) %>%
  ggplot(aes(x=k, y=revenue, color=gran, group=gran)) +
  geom_point() + 
  geom_line() +
  ggtitle("LA Health")
ggsave("fig/la_health_k_rev_by_gran.png")


# k to revenue, faceted by granularity
la_fash %>% 
  group_by(gran, k) %>% 
  summarize(revenue = sum(2*a1*result + 1*(1-a1))) %>% 
  ungroup() %>%
  mutate(gran = paste("gran = ", gran)) %>%
  ggplot(aes(x=k, y=revenue, group=gran)) +
  geom_point() + 
  geom_line() +
  facet_wrap(~ gran) +
  ggtitle("LA Health")
ggsave("fig/la_fash_k_rev_facet_gran.png")

la_travel %>% 
  group_by(gran, k) %>% 
  summarize(revenue = sum(2*a1*result + 1*(1-a1))) %>% 
  ungroup() %>%
  mutate(gran = paste("gran = ", gran)) %>%
  ggplot(aes(x=k, y=revenue, group=gran)) +
  geom_point() + 
  geom_line() +
  facet_wrap(~ gran) +
  ggtitle("LA Travel")
ggsave("fig/la_travel_k_rev_facet_gran.png")

la_health %>% 
  group_by(gran, k) %>% 
  summarize(revenue = sum(2*a1*result + 1*(1-a1))) %>% 
  ungroup() %>%
  mutate(gran = paste("gran = ", gran)) %>%
  ggplot(aes(x=k, y=revenue, group=gran)) +
  geom_point() + 
  geom_line() +
  facet_wrap(~ gran) +
  ggtitle("LA Health")
ggsave("fig/la_health_k_rev_facet_gran.png")


# Difference in a1 across groups by gran
la_fash %>% 
  mutate(gran = as.factor(gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1)) %>% 
  summarize(a1_diff = abs(first(avg_a1) - last(avg_a1))) %>%
  ggplot(aes(x=k, y=a1_diff, color=gran, group=gran)) +
  geom_point() + 
  geom_line() +
  scale_y_continuous("Abs diff prob targeted btwn races") +
  ggtitle("LA Fashion")
ggsave("fig/la_fash_k_a1diff_by_gran.png")

la_travel %>% 
  mutate(gran = as.factor(gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1)) %>% 
  summarize(a1_diff = abs(first(avg_a1) - last(avg_a1))) %>%
  ggplot(aes(x=k, y=a1_diff, color=gran, group=gran)) +
  geom_point() + 
  geom_line() +
  scale_y_continuous("Abs diff prob targeted btwn races") +
  ggtitle("LA Travel")
ggsave("fig/la_travel_k_a1diff_by_gran.png")

la_health %>% 
  mutate(gran = as.factor(gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1)) %>% 
  summarize(a1_diff = abs(first(avg_a1) - last(avg_a1))) %>%
  ggplot(aes(x=k, y=a1_diff, color=gran, group=gran)) +
  geom_point() + 
  geom_line() +
  scale_y_continuous("Abs diff prob targeted btwn races") +
  ggtitle("LA Health")
ggsave("fig/la_health_k_a1diff_by_gran.png")


# Difference in a1 by groups
la_fash %>% 
  mutate(gran = paste("gran = ", gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1)) %>% 
  summarize(a1_diff = abs(first(avg_a1) - last(avg_a1))) %>%
  ggplot(aes(x=k, y=a1_diff, group=gran)) +
  geom_point() + 
  geom_line() +
  facet_wrap(~gran) +
  scale_y_continuous("| Pr(targeted | white) - Pr(targeted | minority) |") +
  ggtitle("LA Fashion")
ggsave("fig/la_fash_k_a1diff_facet_gran.png")

la_travel %>% 
  mutate(gran = paste("gran = ", gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1)) %>% 
  summarize(a1_diff = abs(first(avg_a1) - last(avg_a1))) %>%
  ggplot(aes(x=k, y=a1_diff, group=gran)) +
  geom_point() + 
  geom_line() +
  facet_wrap(~gran) +
  scale_y_continuous("| Pr(targeted | white) - Pr(targeted | minority) |") +
  ggtitle("LA Travel")
ggsave("fig/la_travel_k_a1diff_facet_gran.png")

la_health %>% 
  mutate(gran = paste("gran = ", gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1)) %>% 
  summarize(a1_diff = abs(first(avg_a1) - last(avg_a1))) %>%
  ggplot(aes(x=k, y=a1_diff, group=gran)) +
  geom_point() + 
  geom_line() +
  facet_wrap(~gran) +
  scale_y_continuous("| Pr(targeted | white) - Pr(targeted | minority) |") +
  ggtitle("LA Health")
ggsave("fig/la_health_k_a1diff_facet_gran.png")


# Average a1 by race faceted by granularity
la_fash %>% 
  mutate(gran = paste("gran = ", gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1),
            std_err = sd(a1) / sqrt(n())) %>% 
  ggplot(aes(x=k, y=avg_a1, color=race, group=race)) +
  geom_point() + 
  geom_line() +
  geom_errorbar(aes(ymin=avg_a1 - std_err, ymax=avg_a1 + std_err),
                width=0) +
  facet_wrap(~gran) +
  scale_y_continuous("Average prob. targeted") +
  ggtitle("LA Fashion")
ggsave("fig/la_fash_k_a1_facet_gran_by_race.png")

la_travel %>% 
  mutate(gran = paste("gran = ", gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1),
            std_err = sd(a1) / sqrt(n())) %>% 
  ggplot(aes(x=k, y=avg_a1, color=race, group=race)) +
  geom_point() + 
  geom_line() +
  geom_errorbar(aes(ymin=avg_a1 - std_err, ymax=avg_a1 + std_err),
                width=0) +
  facet_wrap(~gran) +
  scale_y_continuous("Average prob. targeted") +
  ggtitle("LA Travel")
ggsave("fig/la_travel_k_a1_facet_gran_by_race.png")

la_health %>% 
  mutate(gran = paste("gran = ", gran)) %>%
  group_by(gran, k, race) %>% 
  summarize(avg_a1 = mean(a1),
            std_err = sd(a1) / sqrt(n())) %>% 
  ggplot(aes(x=k, y=avg_a1, color=race, group=race)) +
  geom_point() + 
  geom_line() +
  geom_errorbar(aes(ymin=avg_a1 - std_err, ymax=avg_a1 + std_err),
                width=0) +
  facet_wrap(~gran) +
  scale_y_continuous("Average prob. targeted") +
  ggtitle("LA Health")
ggsave("fig/la_health_k_a1_facet_gran_by_race.png")

