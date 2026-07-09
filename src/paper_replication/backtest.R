library(highfrequency)
library(data.table)
library(xts)

# -------------------------------------------------------------------------
# Parameter Estimation
# -------------------------------------------------------------------------

estimate_parameters <- function(t_data, q_data) {

  # estimate Volatility (sigma)
  rv <- rCov(
    t_data[, list(DT, PRICE)],
    align.by = "minutes",
    align.period = 5,
    makeReturns = TRUE
  )

  if ("RV" %in% names(rv)) {
    daily_return_vol <- mean(sqrt(rv$RV), na.rm = TRUE)
  } else {
    daily_return_vol <- mean(sqrt(as.numeric(unlist(rv))), na.rm = TRUE)
  }

  avg_price <- median(as.numeric(t_data$PRICE), na.rm = TRUE)
  sigma_param <- daily_return_vol * avg_price

  # match Trades to Quotes to see where execution ACTUALLY happens
  tq <- matchTradesQuotes(t_data, q_data)
  tq$mid <- (tq$BID + tq$OFR) / 2
  tq$delta <- abs(tq$PRICE - tq$mid)
  valid_tq <- tq[tq$delta > 0 & !is.na(tq$delta)]

  # estimate Kappa using the TOP of the Limit Order Book (10th percentile)
  effective_spreads <- 2 * valid_tq$delta

  # target the tightest 10% of volume to ensure top-of-book queue position
  base_spread <- quantile(effective_spreads, 0.10, na.rm = TRUE)

  # safety rails to prevent negative spread or zero spread which would break the model
  if (is.na(base_spread) || base_spread <= 0) base_spread <- 0.01

  kappa_est <- 2/ base_spread

  # estimate order Arrival intensity (A) constrained by our trade-driven kappa
  valid_tq$bin <- round(valid_tq$delta, 2)
  counts <- table(valid_tq$bin)

  t_data_xts <- as.xts(t_data)
  total_time <- as.numeric(difftime(end(t_data_xts), start(t_data_xts), units = "secs"))

  lambda_df <- data.frame(delta = as.numeric(names(counts)),
                          lambda = as.numeric(counts) / total_time)

  log_A_estimates <- log(lambda_df$lambda) + (kappa_est * lambda_df$delta)
  valid_log_A <- log_A_estimates[is.finite(log_A_estimates)]
  A_est <- exp(median(valid_log_A, na.rm = TRUE))

  # more safety rails to prevent zero or negative A
  if (is.na(sigma_param) || sigma_param == 0) sigma_param <- 1.0
  if (is.na(A_est) || A_est == 0) A_est <- 1.0

  return(list(sigma = sigma_param, A = A_est, kappa = kappa_est))
}

# -------------------------------------------------------------------------
# Simulation Engine
# -------------------------------------------------------------------------

run_simulation <- function(agent, t_data, q_data) {
  # merge Quotes and Trades into a single timeline of events
  quotes_df <- as.data.table(q_data)
  quotes_df[, type := "quote"]
  setnames(quotes_df, c("DT", "BID", "OFR"), c("dt", "bid", "ask"))

  trades_df <- as.data.table(t_data)
  trades_df[, type := "trade"]
  setnames(trades_df,
           c("DT", "PRICE", "SIZE"),
           c("dt", "price", "size"))

  # combined events stream, sorted by time
  events <- rbind(quotes_df[, .(dt, type, bid, ask)], trades_df[, .(dt, type, price, size)], fill = TRUE)
  setkey(events, dt)

  # results storage
  n_events <- nrow(events)
  history_wealth <- numeric(n_events)
  history_inv    <- numeric(n_events)
  history_mid    <- numeric(n_events)
  history_bid    <- numeric(n_events) # Agent's bid
  history_ask    <- numeric(n_events) # Agent's ask

  # simulation state
  current_mid <- NA
  start_time <- as.numeric(events$dt[1])

  # assume T=1 is the full session, per the classic A-S implementation
  session_duration <- as.numeric(difftime(events$dt[length(events$dt)], events$dt[1], units="secs"))


  for (i in 1:n_events) {
    row <- events[i]
    time_elapsed <- as.numeric(row$dt) - start_time

    # normalize time t to  for the formula
    t_normalized <- time_elapsed / session_duration

    if (row$type == "quote") {
      # update mid price
      current_mid <- (row$bid + row$ask) / 2

      # agent updates our quotes
      my_quotes <- agent$get_quotes(current_mid, t_normalized)

    } else if (row$type == "trade" && !is.na(current_mid)) {
      # a market trade occurred. did it hit our quotes?

      # assumption: we represent a small part of liquidity.
      # if market trade price <= agent bid, agent limit buy quote filled
      if (!is.na(agent$current_bid) &&
          row$price <= agent$current_bid) {
        agent$fill_order(agent$current_bid, 1, 1) # buy 1 unit
      }

      # if market trade price >= agent ask/offer, agent limit sell is filled
      if (!is.na(agent$current_ask) &&
          row$price >= agent$current_ask) {
        agent$fill_order(agent$current_ask, 1, -1) # sell 1 unit
      }
    }

    # record history
    history_wealth[i] <- agent$get_wealth(current_mid)
    history_inv[i]    <- agent$inventory
    history_mid[i]    <- current_mid
    history_bid[i]    <- agent$current_bid
    history_ask[i]    <- agent$current_ask
  }

  return(
    data.table(
      dt = events$dt,
      mid = history_mid,
      wealth = history_wealth,
      inventory = history_inv,
      agent_bid = history_bid,
      agent_ask = history_ask
    )
  )
}
