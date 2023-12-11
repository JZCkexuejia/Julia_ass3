using LinearAlgebra

"""
Compute the integral ∫f(x)dx over [a, b] with the composite trapezoidal
rule using r subintervals.

Inputs:
    f: function to integrate
    a: lower bound of the definite integral
    b: upper bound of the definite integral
    r: number of subintervals
"""
function composite_trapezoidal_rule(f, a, b, r)
    h = (b - a) / r
    sum = 0

    for i in 1:r-1
        sum += f(a + i * h)
    end

    res = 0.5 * h * (f(a) + f(b)) + h * sum

    return res
end

"""
Compute the integral ∫f(x)dx over [a, b] with the composite midpoint 
rule using r subintervals.

Inputs:
    f: function to integrate
    a: lower bound of the definite integral
    b: upper bound of the definite integral
    r: number of subintervals
"""
function composite_midpoint_rule(f, a, b, r)
    h = (b - a) / r
    res = 0

    for i in 1:r
        l = a + (i - 1) * h
        r = a + i * h
        res += (r - l) * f((l + r) / 2)
    end

    return res
end

"""
Compute the integral ∫f(x)dx over [a, b] with the composite Simpson's 
rule using r subintervals. Note that r must be even because each 
application of Simpson's rule uses a subinterval of length 2*(b-a)/r.
In other words, the midpoints used by the basic Simpson's rule are 
included in the r+1 points on which we evaluate f(x).

Inputs:
    f: function to integrate
    a: lower bound of the definite integral
    b: upper bound of the definite integral
    r: even number of subintervals
"""
function composite_simpsons_rule(f, a, b, r)
    h = 2 * (b - a) / r
    res = 0

    for i in 1:r/2
        l = a + (i - 1) * h
        r = a + i * h

        res += h / 6 * (f(l) + 4 * f((l + r) / 2) + f(r))
    end

    return res
end

"""
Compute the integral ∫f(x)dx over [a, b] with the adaptive Simpson's 
rule. Return the approximate integral along with the nodes (points) x 
used to compute it.  

Inputs:
    f: function to integrate
    a: lower bound of the definite integral
    b: upper bound of the definite integral
    tol: maximum error we can approximately tolerate (i.e., |If - Q| <≈ tol)
    max_depth: maximum number of times this function should be recursively called

Returns:
    approximate_integral: the value of the integral ∫f(x)dx over [a, b]
    x: vector containing the nodes which the algorithm used to compute approximate_integral
"""
function adaptive_simpsons_rule(f, a, b, tol, max_depth)
    m = (a + b) / 2

    S(l, r) = 1 / 6 * (r - l) * (f(l) + 4 * f((l + r) / 2) + f(r))

    s_curr = S(a, b)
    s_next = S(a, m) + S(m, b)

    if abs(s_next - s_curr) / 15 <= tol || max_depth == 0
        return s_next, [a, m, b]
    else
        left_integral, left_nodes = adaptive_simpsons_rule(f, a, m, tol / 2, max_depth - 1)
        right_integral, right_nodes = adaptive_simpsons_rule(f, m, b, tol / 2, max_depth - 1)

        return left_integral + right_integral, vcat(left_nodes, right_nodes[2:end]) # Avoid duplication of midpoint
    end
end

"""
Use Newton's method to solve the nonlinear system of equations described in Problem 5.
This should work for Euclidean distance measurements in any dimension n.

Inputs:
    x0: initial guess for the position of the receiver in R^n
    P: nxn matrix with known locations of transmitting beacons as columns
    d: vector in R^n where d[i] contains the distance from beacon P[:, i] to x
    tol: Euclidean error tolerance (stop when norm(F(x)) <= tol)
    max_iters: maximum iterations of Newton's method to try

Returns:
    x_trace: Vector{Vector{Float64}} containing each Newton iterate x_k in R^n. 

"""
function newton(x0, P, d, tol, max_iters)
    N = length(x0)
    x_trace = [x0]

    F = [norm(x0 - P[:, i]) - d[i] for i in 1:N]

    for _ in 1:max_iters
        J = zeros((N, N))

        for i in 1:N
            for j in 1:N
                J[i, j] = (x_trace[end][j] - P[j, i]) / norm(x_trace[end] - P[:, i])
            end
        end

        push!(x_trace, x_trace[end] - J \ F)

        F = [norm(x_trace[end] - P[:, i]) - d[i] for i in 1:N]

        if norm(F) <= tol
            break
        end
    end

    return x_trace
end

"""
Use Newton's method to solve the nonlinear optimization problem described in Problem 7.
This should work for Euclidean distance measurements in any dimension n, and any number 
    of noisy measurements m.

Inputs:
    x0: initial guess for the position of the receiver in R^n
    P: nxm matrix with known locations of transmitting beacons as columns
    d: vector in R^m where d[i] contains the noisy distance from beacon P[:, i] to x
    tol: Euclidean error tolerance (stop when norm(∇f(x)) <= tol)
    max_iters: maximum iterations of Newton's method to try

Returns:
    x_trace: Vector{Vector{Float64}} containing each Newton iterate x_k in R^n. 

"""
function gradient(x, P, d)
    N, M = length(x), length(d)
    grad = zeros(N)

    for i in 1:N
        for m in 1:M
            grad[i] += (norm(x - P[:, m]) - d[m]) * (x[i] - P[i, m]) / (norm(x - P[:, m]))
        end
    end

    return 2 * grad
end

function hessian(x, P, d)
    N, M = length(x), length(d)
    hess = zeros((N, N))

    for i in 1:N
        for j in 1:i # Lower triangle only
            for m in 1:M
                if i == j
                    hess[i, j] += ((norm(x - P[:, m]) - d[m]) / norm(x - P[:, m])
                                   +
                                   ((x[i] - P[i, m]) / norm(x - P[:, m]))^2
                                   -
                                   (norm(x - P[:, m]) - d[m]) * (x[i] - P[i, m])^2 / norm(x - P[:, m])^3)
                else
                    hess[i, j] += -(norm(x - P[:, m]) - d[m]) * (x[i] - P[i, m]) * (x[j] - P[j, m]) / norm(x - P[:, m])^3 + (x[i] - P[i, m]) * (x[j] - P[j, m]) / norm(x - P[:, m])^2
                    hess[j, i] = hess[i, j] # Copy lower triangle to upper triangle
                end
            end
        end
    end

    return 2 * hess
end

function newton_optimizer(x0, P, d, tol, max_iters)
    x_trace = [x0]

    grad = gradient(x0, P, d)
    hess = hessian(x0, P, d)

    for _ in 1:max_iters
        push!(x_trace, x_trace[end] - hess \ grad)

        grad = gradient(x_trace[end], P, d)
        hess = hessian(x_trace[end], P, d)

        if norm(grad) <= tol
            break
        end
    end

    return x_trace
end