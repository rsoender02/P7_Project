library(MASS)   # for polynomial regression
library(ggplot2)
# Parameters
r <- 0.1         # Risk-free rate
S0 <- 100        # Initial stock price
J <- 252         # Number of time steps (exercise dates)
k <- 100000       # Number of training trajectories
sigma <- 0.1     # Volatility
K <- 100         # Strike price
T <- 1           # Maturity (1 year)
dt <- T / J      # Time step size
discount <- exp(-r * dt)  # Discount factor
u <- exp(sigma * sqrt(dt))         # Up factor
d <- 1/u
p <- (exp(r * dt) - d) / (u - d)   # Risk-neutral probability

# Simulate asset paths using the Black-Scholes model
set.seed(2)
Z <- matrix(rnorm(k * J), nrow = k, ncol = J)
S <- matrix(0, nrow = k, ncol = J + 1) 
S[, 1] <- S0

# Generate paths
for (j in 1:J) {
  S[, j + 1] <- S[, j] * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z[, j])
}

# matplot(seq(J+1), t(S[1:10,]), type = "l", lwd = 1, lty = 1, xlab = expression(italic("t")), ylab = "Simulated stock prices", main = NULL)

# Initialize payoff and value matrices
payoff <- pmax(S - K, 0)  # Payoff for a put option
cashflow <- payoff[, J + 1] * discount # Terminal payoff at maturity discounted
ex <- rep(J+1, k)

# Backward induction
for (j in J:1) {
  itm <- which(payoff[, j] > 0)  # Indices of in-the-money paths
  if (length(itm) > 0) {
    # Regression to estimate continuation value
    X <- S[itm, j]
    Y <- cashflow[itm]
    reg <- lm(Y ~ poly(S[itm, j], 2, raw = TRUE))  # Polynomial regression
    continuation <- predict(reg, newdata = data.frame(X = S[itm, j]))
    
    # Exercise decision: Max(payoff, continuation)
    exercise <- payoff[itm, j] > continuation
    cashflow[itm] <- ifelse(exercise, payoff[itm, j], cashflow[itm])
    ex[itm[exercise]] <- j
  }
  cashflow <- cashflow * discount
}

# Calculate option price
price <- mean(cashflow)
cat("American Option Price:", price, "\n")

# Prepare and plot histogram
ggplot(data.frame(
  values = ex,
  category = factor(ifelse(payoff[cbind(1:k, ex)] > 0, "In The Money", "Out of the money"),
                    levels = c("Out of the money", "In The Money"))
), aes(x = values, fill = category)) +
  xlim(0, 260) +
  geom_histogram(bins = 30, position = "stack", alpha = 0.7) +
  scale_fill_manual(
    values = c("Out of the money" = "red", "In The Money" = "black"),
    breaks = c("Out of the money") # Show only "Out of the money" in the legend
  ) +
  theme_minimal() +
  theme(
    legend.position = c(0.25, 0.95), # Adjusted to the top-center
    legend.justification = c("center", "top"), # Adjust to be anchored at the top-center
    legend.key.size = unit(0.8, "cm"), # Adjust size of legend key
    legend.text = element_text(size = 8), # Adjust legend text size
    legend.title = element_blank(), # Remove legend title
    legend.spacing.y = unit(0.1, "cm"), # Adjust vertical spacing between legend items
    legend.spacing.x = unit(0.1, "cm"), # Adjust horizontal spacing (if needed)
    legend.background = element_rect(
      fill = "white",        # Background color for the legend
      color = "black",       # Border color
      size = 0.3,            # Border thickness
      linetype = "solid"     # Border line type
    )
  ) +
  labs(
    x = "Estimated optimal stopping time",
    y = "Frequency"
  )
table(ex)
sum(ex<253)/k*100
analytical <- 0
for (j in 0:J){
  analytical <- analytical + choose(J, j)*p^j*(1-p)^(J-j)*max(u^j*d^(J-j)*S0-K, 0)*exp(-r*dt)^J
}
analytical


results_matrix_clean <- data.frame(t(read.csv("path-to-file")))

results_matrix_clean <- results_matrix_clean[2:nrow(results_matrix_clean),5:ncol(results_matrix_clean)]

matplot(seq(100), results_matrix_clean, type = "l", lwd = 1, lty = 1, xlab = expression(k %*% 10^4), col = c("#DF536B", "#61D04F", "#2297E6"), ylab = "Estimation error (%)", main = NULL, ylim = c(-1, 1))

abline(h = 0, lwd = 0.5, lty = 2)

legend(
  "topright",                       # Position of the legend
  legend = c("Seed 1", "Seed 2", "Seed 3"), # Seed labels
  col = c("#DF536B", "#61D04F", "#2297E6"),     # Colors matching the matplot
  lty = 1,                          # Line type
  lwd = 1,                          # Line width
  cex = 0.5
)

