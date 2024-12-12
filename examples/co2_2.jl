using CairoMakie
using Optim
using StatsBase
using LinearAlgebra
using DelimitedFiles
using Celerite
using Celerite: SHOTerm, CeleriteGP, compute!, log_likelihood, get_psd

"""
Compute log-likelihood for quasi-periodic GP model.
"""
function neg_log_like(param, t, y, yerr)
    # Extract parameters
    mean_flux = param[1]
    # GP hyperparameters for quasi-periodic kernel
    log_amp = param[2]  # log amplitude
    log_period = param[3]  # log period
    log_q = param[4]  # log Q factor
    log_mix = param[5]  # log mixture parameter for periodic vs decay

    # Create kernel: quasi-periodic = A * exp(-τ/τ_d) * exp(-sin²(πτ/P)/(2Q²))
    kernel = SHOTerm(log_amp, log_q, log_period)
    
    # Create GP object
    gp = CeleriteGP(kernel)

    # Compute GP model
    compute!(gp, t, yerr) 
    # Compute log likelihood
    nll = -log_likelihood(gp, y .- mean_flux)
    
    return nll
end

"""
Analyze CO2 data with quasi-periodic GP model.
"""
function analyze_co2_qp()
    # Read data
    data = readdlm("data/CO2_data.csv", ',', comment_char='#', comments=true)
    co2 = Vector{Float64}(vec(data[:, 2]))
    time = Vector{Float64}(vec(data[:, 1] .- data[1, 1]))
    
    # Remove quadratic trend first
    nt = length(time)
    ord = 2
    fn = zeros(ord + 1, nt)
    tnorm = (time .- time[1]) ./ (time[nt] - time[1])
    for i in 1:ord+1
        fn[i, :] .= tnorm.^(i-1)
    end
    
    # Compute regression
    coeff = fn' \ co2
    co2_trend = fn' * coeff
    co2_sub = co2 .- co2_trend
    
    # Initial parameters
    param_init = [
        0.0,            # mean flux
        log(var(co2_sub)),  # log amplitude
        log(1.0),      # log period (1 year)
        log(10.0),     # log Q factor
        log(0.5),      # log mixture
    ]

    # Parameter bounds
    lower = [-5.0, -10.0, -1.0, 0.0, -5.0]
    upper = [5.0, 10.0, 2.0, 5.0, 0.0]

    co2_sub_err = 0.1 .+ zeros(length(co2_sub))# assume uniform measurement uncertainties for now

    # Optimize GP parameters
    obj = x -> neg_log_like(x, time, co2_sub, co2_sub_err)
    # res = optimize(obj, lower, upper, param_init, Fminbox(LBFGS()), Optim.Options(show_trace=true))#, autodiff = :forward)
    # res = optimize(obj, lower, upper, param_init, Fminbox(LBFGS()), Optim.Options(show_trace=true))#, autodiff = :forward)
    res = optimize(obj, param_init, (NelderMead()), Optim.Options(show_trace=true))
    param_opt = Optim.minimizer(res)
    # param_opt = param_init
    
    # Create optimized kernel
    kernel_opt = SHOTerm(param_opt[2], param_opt[4], param_opt[3])
    gp_opt = CeleriteGP(kernel_opt)

    # Create time grid for smooth predictions
    time_pred = range(minimum(time), maximum(time) + 5, length=10length(time))
    
    # Compute GP on data points
    @time "compute" compute!(gp_opt, time, zeros(length(time)))
    
    # Compute GP prediction on both original and interpolated points
    @time "predict" pred_mean, pred_var = Celerite.predict(gp_opt, co2_sub, collect(time_pred); return_var=true)
    # pred_var .= 0
    pred_var = max.(0, pred_var)
    pred_std = sqrt.(pred_var)
    
    # Plot results
    fig = Figure(size=(900, 600))
    
    ax1 = Axis(fig[1, 1], xlabel="Time", ylabel="CO2 (detrended)")
    # Plot smooth prediction
    lines!(ax1, time_pred, pred_mean, color=:blue, label="GP mean")
    # Plot data points
    scatter!(ax1, time, co2_sub, color=:black, markersize=4, label="Data")
    # Plot uncertainty band
    band!(ax1, time_pred, pred_mean .- 2pred_std, pred_mean .+ 2pred_std, 
          color=(:blue, 0.2), label="2σ")
    axislegend()
    
    # Compute and plot power spectrum
    omega = exp.(range(log(0.1), log(20), length=1000))
    psd = Celerite.get_psd2(kernel_opt, omega)
    
    ax2 = Axis(fig[2, 1], xlabel="Frequency", ylabel="Power", yscale=log10)
    lines!(ax2, omega, psd)

    display(fig)
    
    return time, time_pred, co2_sub, pred_mean, pred_std, param_opt
end

# Run analysis
time, time_pred, data, mean_pred, std_pred, params = analyze_co2_qp()