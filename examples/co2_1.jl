using CairoMakie
using StatsBase
using LinearAlgebra
using DelimitedFiles

function analyze_co2()
    # Read in CO2 data
    data = readdlm("data/CO2_data.csv", ',',comment_char='#',comments=true)
    co2 = Vector{Float64}(vec(data[:, 2]))
    time = Vector{Float64}(vec(data[:, 1] .- data[1, 1]))
    
    # Fit a quadratic trend
    nt = length(time)
    ord = 2
    fn = zeros(ord + 1, nt)
    tnorm = (time .- time[1]) ./ (time[nt] - time[1])
    
    # Build design matrix
    for i in 1:ord+1
        fn[i, :] .= tnorm.^(i-1)
    end
    
    # Compute regression using direct solve
    coeff = fn' \ co2  # This is more numerically stable
    
    # Compute trend
    co2_trend = fn' * coeff
    co2_sub = co2 .- co2_trend
    
    # Plot results
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Time", ylabel="CO2")
    lines!(ax, time, co2, label="Data")
    lines!(ax, time, co2_trend, label="Trend")
    lines!(ax, time, co2_sub, label="Detrended")
    axislegend()
    display(fig)
    
    return time, co2_sub
end

# Run the analysis
time, detrended = analyze_co2()