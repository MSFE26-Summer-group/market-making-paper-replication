# Parameter Sensitivity Sweep for Gamma
# Uses the estimated kappa from data (not overridden).
wd <-getwd()
setwd("./avellaneda-stoikov/")

source("A-S_strategy.R")
source("backtest.R")

require(highfrequency)

data("sampleTData", package = "highfrequency")
data("sampleQData", package = "highfrequency")
t_subset <- sampleTData
q_subset <- sampleQData

# Calibrate base parameters
params <- estimate_parameters(t_subset, q_subset)
cat(sprintf("Estimated: sigma=%.4f, A=%.4f, kappa=%.4f\n",
            params$sigma, params$A, params$kappa))

# Sweep gamma and kappa, centering kappa around the estimated value
gamma_values <- c(0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
kappa_values <- c(5, 10, 20, 30, 40, 52, 75, 100)

# Results table
sweep_results <- expand.grid(gamma = gamma_values, kappa = kappa_values)
sweep_results$final_pnl   <- NA
sweep_results$max_inv     <- NA
sweep_results$avg_spread  <- NA

cat(sprintf("Running sweep over %d combinations (estimated kappa=%.2f)...\n",
            nrow(sweep_results), params$kappa))

for (i in 1:nrow(sweep_results)) {
  g <- sweep_results$gamma[i]
  k <- sweep_results$kappa[i]

  agent <- AvellanedaStoikovAgent$new(
    gamma = g,
    sigma = params$sigma,
    kappa = k,
    A = params$A,
    T_session = 1.0,
    initial_cash = 100000
  )

  res <- run_simulation(agent, t_subset, q_subset)

  sweep_results$final_pnl[i]  <- tail(res$wealth, 1) - 100000
  sweep_results$max_inv[i]    <- max(abs(res$inventory), na.rm = TRUE)
  sweep_results$avg_spread[i] <- mean(res$agent_ask - res$agent_bid, na.rm = TRUE)

  cat(sprintf("  gamma=%-6s kappa=%-3s => PnL=%10.2f  MaxInv=%6d  AvgSpread=%.4f\n",
              g, k, sweep_results$final_pnl[i],
              sweep_results$max_inv[i], sweep_results$avg_spread[i]))
}

# Reshape PnL into a matrix for heatmap
pnl_matrix <- matrix(sweep_results$final_pnl,
                      nrow = length(gamma_values),
                      ncol = length(kappa_values),
                      byrow = FALSE)
rownames(pnl_matrix) <- gamma_values
colnames(pnl_matrix) <- kappa_values

cat("\n=== PnL Heatmap (rows=gamma, cols=kappa) ===\n")
print(round(pnl_matrix, 2))

# Reshape MaxInv into a matrix
inv_matrix <- matrix(sweep_results$max_inv,
                     nrow = length(gamma_values),
                     ncol = length(kappa_values),
                     byrow = FALSE)
rownames(inv_matrix) <- gamma_values
colnames(inv_matrix) <- kappa_values

cat("\n=== Max Inventory Heatmap (rows=gamma, cols=kappa) ===\n")
print(inv_matrix)

# Reshape AvgSpread into a matrix
spread_matrix <- matrix(sweep_results$avg_spread,
                        nrow = length(gamma_values),
                        ncol = length(kappa_values),
                        byrow = FALSE)
rownames(spread_matrix) <- gamma_values
colnames(spread_matrix) <- kappa_values

cat("\n=== Avg Spread Heatmap (rows=gamma, cols=kappa) ===\n")
print(round(spread_matrix, 4))

# Plot
par(mfrow = c(1, 1))
image(
  x = log10(gamma_values),
  y = kappa_values,
  z = pnl_matrix,
  xlab = "log10(gamma)",
  ylab = "kappa",
  main = "Final PnL: Gamma vs Kappa Sensitivity",
  col = hcl.colors(20, "RdYlGn")
)
contour(
  x = log10(gamma_values),
  y = kappa_values,
  z = pnl_matrix,
  add = TRUE
)

setwd(wd)
