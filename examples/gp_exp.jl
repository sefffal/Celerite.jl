using CairoMakie
using Random
using LinearAlgebra

function simulate_exp_gp_final(t::Vector{Float64}, alpha::Complex{Float64}, 
                             beta::Complex{Float64}, ndev::Vector{Float64})
    nt = length(t)
    data = zeros(Complex{Float64}, nt)
    
    # Initialize first point with correct variance
    data[1] = sqrt(real(alpha)) * ndev[1]
    
    # Forward simulation with corrected normalization
    for i in 2:nt
        dt = t[i] - t[i-1]
        
        # Complex correlation at this timestep
        gamma = exp(-beta * dt)
        
        # Theoretical covariance at this lag
        cov_dt = exp(-real(beta)*dt) * 
                 (real(alpha)*cos(imag(beta)*dt) + imag(alpha)*sin(imag(beta)*dt))
        
        # Calculate innovation variance to maintain theoretical marginal variance
        target_var = real(alpha)  # Target marginal variance
        innovation_var = target_var * (1.0 - abs2(cov_dt/target_var))
        
        # Generate new point
        data[i] = gamma * data[i-1] + sqrt(complex(innovation_var)) * ndev[i]
    end
    
    return data
end

function run_simulation()
    # Set up parameters
    nt = 100
    t = collect(range(0, 10, length=nt))
    
    # Complex parameters
    beta = 0.05 + 1.0im
    alpha = 1.0 + 0.0im
    
    # Generate random deviates
    Random.seed!(42)
    ndev = randn(nt)
    
    # Simulate GP using final method
    data_final = simulate_exp_gp_final(t, alpha, beta, ndev)
    
    # Build exact covariance matrix for comparison
    K = zeros(Complex{Float64}, nt, nt)
    for i in 1:nt
        for j in 1:nt
            dt = abs(t[i] - t[j])
            K[i,j] = exp(-real(beta)*dt) * 
                     (real(alpha)*cos(imag(beta)*dt) + imag(alpha)*sin(imag(beta)*dt))
        end
    end
    
    # Generate comparison data using Cholesky
    L = cholesky(Hermitian(K)).L
    sim_exact = L * ndev
    
    # Check covariances at multiple lags
    println("Variance comparison:")
    println("Theoretical variance: ", real(alpha))
    println("Exact simulation variance: ", var(real(sim_exact)))
    println("Final simulation variance: ", var(real(data_final)))
    
    println("\nAutocorrelation comparison:")
    for lag in 1:3
        ac_exact = cor(real(sim_exact[1:end-lag]), real(sim_exact[1+lag:end]))
        ac_final = cor(real(data_final[1:end-lag]), real(data_final[1+lag:end]))
        println("Lag-$lag autocorrelation:")
        println("  Exact: ", ac_exact)
        println("  Final: ", ac_final)
    end
    
    # Plotting
    fig = Figure(size=(900, 900))
    
    ax1 = Axis(fig[1, 1], xlabel="Time", ylabel="Real Part")
    lines!(ax1, t, real(sim_exact), label="Exact", color=:black, linestyle=:dash)
    lines!(ax1, t, real(data_final), label="Final", color=:blue)
    axislegend(ax1)
    
    ax2 = Axis(fig[2, 1], xlabel="Time", ylabel="Difference")
    lines!(ax2, t, real(data_final - sim_exact), 
          label="Final - Exact", color=:red)
    axislegend(ax2)
    
    # Add autocorrelation plot
    max_lag = 20
    ac_exact = [cor(real(sim_exact[1:end-k]), real(sim_exact[1+k:end])) 
                for k in 0:max_lag]
    ac_final = [cor(real(data_final[1:end-k]), real(data_final[1+k:end])) 
                for k in 0:max_lag]
    
    ax3 = Axis(fig[3, 1], xlabel="Lag", ylabel="Autocorrelation")
    lines!(ax3, 0:max_lag, ac_exact, label="Exact", color=:black, linestyle=:dash)
    lines!(ax3, 0:max_lag, ac_final, label="Final", color=:blue)
    axislegend(ax3)
    
    display(fig)
    
    return t, data_final, sim_exact, K
end

# Run simulation
t, final_gp, exact_gp, K = run_simulation()