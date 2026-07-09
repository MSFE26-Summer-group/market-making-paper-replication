# Load sources
wd <-getwd()
setwd("./avellaneda-stoikov/")
source("A-S_strategy.R")
source("backtest.R")
# source("test_strategy.R") # Run unit tests

require(highfrequency)

# load data
# use sample data from highfrequency package
data("sampleTData", package = "highfrequency")
data("sampleQData", package = "highfrequency")

# take a subset (e.g., one trading day) to speed up demo
#t_subset <- sampleTData["2018-01-03",]
#q_subset <- sampleQData["2018-01-03",]
t_subset <- sampleTData
q_subset <- sampleQData

# estimate parameters
message("Calibrating parameters...")
params <- estimate_parameters(t_subset, q_subset)
# print(paste("Estimated A:", round(params$A, 2)))
# print(paste("Estimated Kappa:", round(params$kappa, 2)))
# print(paste("Estimated Sigma:", round(params$sigma, 4)))

# initialize agent
# normalize T_session to 1.0 to match the standard AS formula inputs
# gamma is the risk aversion knob
agent <- AvellanedaStoikovAgent$new(
  gamma = 0.05,
  sigma = params$sigma,
  kappa = params$kappa,
  A = params$A,
  T_session = 1.0,
  initial_cash = 100000
)

# run backtest
message("Running Simulation...")
results <- run_simulation(agent, t_subset, q_subset)

# visualization
par(mfrow = c(3, 1))

# plot 1: market prices
plot(
  results$mid,
  type = 'l',
  col = 'gray',
  main = "Market Making: Quotes vs Mid",
  ylab = "Price",
  ylim = c(
    min(results$agent_bid, na.rm = TRUE),
    max(results$agent_ask, na.rm = TRUE)
  )
)
lines(results$agent_bid, col = 'green')
lines(results$agent_ask, col = 'red')

# plot 2: inventory
plot(
  results$inventory,
  type = 'h',
  col = 'blue',
  main = "Inventory Profile",
  ylab = "Position (Units) Held"
)
abline(h = 0, col = "black")

# plot 3: wealth (P&L)
plot(
  results$wealth-100000,
  type = 'l',
  col = 'purple',
  main = "Total Wealth",
  ylab = "P&L (relative to cash)"
)

# summary stats
final_pnl <- tail(results$wealth, 1) - 100000
message(paste("Final PnL:", round(final_pnl, 2)))


setwd(wd)
