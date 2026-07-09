library(testthat)
source("A-S_strategy.R")


test_that("Avellaneda Stoikov Logic", {
  # Setup a standard agent
  # T=1, gamma=0.1, sigma=2, kappa=1.5, A=140
  agent <- AvellanedaStoikovAgent$new(
    gamma = 0.1,
    sigma = 2,
    kappa = 1.5,
    A = 140,
    T_session = 1.0
  )

  # Test 1: Zero Inventory Symmetry
  # If inventory is 0, reservation price should equal mid price
  quotes <- agent$get_quotes(mid_price = 100, current_time = 0)
  expect_equal(agent$reservation_price, 100)

  # Spread should be symmetric around 100
  dist_bid <- 100 - quotes$bid
  dist_ask <- quotes$ask - 100
  expect_equal(dist_bid, dist_ask)

  # Test 2: Long Inventory Skew
  # If inventory > 0, reservation price should decrease (eager to sell)
  agent$inventory <- 1000
  quotes_long <- agent$get_quotes(mid_price = 100, current_time = 0)

  expect_lt(agent$reservation_price, 100)
  expect_lt(quotes_long$bid, quotes$bid) # Bid drops
  expect_lt(quotes_long$ask, quotes$ask) # Ask drops (more aggressive sell)

  # Test 3: Short Inventory Skew
  # If inventory < 0, reservation price should increase (eager to buy)
  agent$inventory <- -1000
  quotes_short <- agent$get_quotes(mid_price = 100, current_time = 0)

  expect_gt(agent$reservation_price, 100)

  # Test 4: Time Convergence
  # As t -> T, the risk component (gamma * sigma^2 * (T-t)) goes to 0
  # The spread should tighten or converge to the asymptotic limit
  agent$inventory <- 0
  quotes_start <- agent$get_quotes(mid_price = 100, current_time = 0)
  quotes_end   <- agent$get_quotes(mid_price = 100, current_time = 0.99)

  spread_start <- quotes_start$ask - quotes_start$bid
  spread_end   <- quotes_end$ask - quotes_end$bid

  # In AS model, spread decreases as T approaches because volatility risk decreases
  expect_lt(spread_end, spread_start)
})
