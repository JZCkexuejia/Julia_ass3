using Plots

include("assignment3_handout.jl")

# Fixed 2D anchor positions
P = [0.0 0.0 100.0 100.0;
     0.0 100.0 0.0 100.0]

# Ground truth location of the receiver
x_gt = [20.0; 30.0]

# Compute the noisy distances between the anchors and the receiver
d = [norm(P[:, i] - x_gt) + randn() for i in 1:size(P, 2)]

# Our initial guess for the location of the receiver
x0 = [40.0; 80.0]

# Run your Newton's method-based optimizer
x_trace = newton_optimizer(x0, P, d, 1e-3, 20)

# Plot the results on a contour map
x1 = [x[1] for x in x_trace]
x2 = [x[2] for x in x_trace]

limit = 100.0
N_plot = 100
X1 = range(-limit, stop=limit, length=N_plot)
X2 = range(-limit, stop=limit, length=N_plot)

# Define the cost function
function f(x, P, d)
    cost = 0.0
    for i in 1:size(P, 2)
        cost += (norm(x - P[:,i]) - d[i])^2
    end
    return cost
end

# Create a modified cost function for ease of 2D plotting
f_plot(x1, x2) = f([x1; x2], P, d)
z1 = @. f_plot(X1', X2)
contourf(X1, X2, log10.(z1), fill=true, color=:turbo,
         lw=0.0, aspect_ratio=:equal, legend=:bottomleft)
plot!(x1, x2, label="Newton Iterations", marker=:o, color=:white)
scatter!([x_gt[1]], [x_gt[2]], label="Ground Truth", marker=:x, color=:red, markersize=9)
scatter!(P[1, :], P[2, :], label="Beacons")
title!("log10 of Cost")
