using Plots

include("assignment3_handout.jl")

# Define the integrand parameterized by k and the integral's bounds
f_pendulum(x, k) = 1/(sqrt(1 - k^2*sin(x)^2))
a = 0.0
b = π/2

# Parameters for the adaptive Simpson's rule
max_depth = 10
tol = 0.001

# Discretize the values of k
n_k = 100
k_vals = range(-1+1e-6, stop=1-1e-6, length=n_k)
int_comp = zeros(n_k)
int_adapt = zeros(n_k)

# Compute approximate integrals for each value of k 
for (i, k) in enumerate(k_vals)
    f_k(x) = f_pendulum(x, k)
    int_adapt_k, x_adapt = adaptive_simpsons_rule(f_k, a, b, tol, max_depth)

    # Use the same number of points the adaptive rule needed for the composite rule
    n_adapt = length(x_adapt)
    int_comp_k = composite_simpsons_rule(f_k, a, b, n_adapt)
    int_adapt[i] = int_adapt_k
    int_comp[i] = int_comp_k
end

p1 = plot(k_vals, int_adapt, label="Adaptive")
plot!(k_vals, int_comp, label="Composite", linestyle=:dash)
title!("Comparing Adaptive and Composite Rules")
xaxis!("k")
yaxis!("T(k)")
display(p1)

# Next, investigate the distribution of points (nodes) for k=0.99
f(x) = f_pendulum(x, 0.99)
int_adapt_k, x_adapt = adaptive_simpsons_rule(f, a, b, tol, max_depth)
n_adapt = length(x_adapt)
int_comp_k = composite_simpsons_rule(f, a, b, n_adapt)
h = (b-a)/n_adapt
x_comp = [a+h*i for i in 0:n_adapt]
n_plot = 1000
x_plot = range(a, stop=b, length=n_plot)
plot(x_plot, f.(x_plot), label="Integrand")
scatter!(x_adapt, zeros(length(x_adapt)), label="Adaptive Nodes")
scatter!(x_comp, zeros(length(x_comp)), label="Composite Nodes", marker=:d, color=:black) 
