library(R6)

AvellanedaStoikovAgent <- R6Class(
  "AvellanedaStoikovAgent",
  public = list(
    ## parameters
    gamma = NULL,
    # risk aversion
    sigma = NULL,
    # volatility (per-period consistent with T)
    kappa = NULL,
    # order book liquidity density
    A = NULL,
    # order arrival intensity
    T_session = NULL,
    # total session duration (in same units as time_elapsed)

    # state
    inventory = 0,
    cash = 0,

    # current quotes (for observation/debugging)
    reservation_price = NA,
    current_bid = NA,
    current_ask = NA,

    initialize = function(gamma,
                          sigma,
                          kappa,
                          A,
                          T_session,
                          initial_cash = 100000) {
      self$gamma <- gamma
      self$sigma <- sigma
      self$kappa <- kappa
      self$A <- A
      self$T_session <- T_session
      self$cash <- initial_cash
    },

    get_quotes = function(mid_price, current_time) {
      # remaining time
      T_left <- max(0, self$T_session - current_time)

      # calculate Reservation Price (r)
      # r = s - q * gamma * sigma^2 * (T - t)
      # adjust the "fair value" based on inventory risk.
      # if q > 0 (long), r < s we lower price to sell.
      self$reservation_price <- mid_price - (self$inventory * self$gamma * (self$sigma^2) * T_left)

      # calculate Optimal Spread (delta)
      # delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)
      # the first term is the risk component, the second is the liquidity component.
      risk_component <- self$gamma * (self$sigma^2) * T_left
      liquidity_component <- (2 / self$gamma) * log(1 + (self$gamma / self$kappa))

      optimal_spread <- risk_component + liquidity_component
      half_spread <- optimal_spread / 2

      # set Bid/Ask around Reservation Price
      self$current_bid <- self$reservation_price - half_spread
      self$current_ask <- self$reservation_price + half_spread

      return(list(bid = self$current_bid, ask = self$current_ask))
    },

    # execute a trade
    # quantity is usually 1 or a standard lot size in simulations
    # side: 1 = Agent BUY, -1 = Agent SELL
    fill_order = function(price, quantity, side) {
      if (side == 1) {
        self$inventory <- self$inventory + quantity
        self$cash <- self$cash - (price * quantity)
      } else {
        self$inventory <- self$inventory - quantity
        self$cash <- self$cash + (price * quantity)
      }
    },

    # Mark-to-Market wealth
    get_wealth = function(mid_price) {
      return(self$cash + self$inventory * mid_price)
    }
  )
)
