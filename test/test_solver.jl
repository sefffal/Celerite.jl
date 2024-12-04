using Random
using LinearAlgebra
@testset "Solver" begin
    Random.seed!(42)
    N = 100
    x = sort(10 .* rand(N))
    y = sin.(x)
    yerr = 0.01 .+ rand(N) ./ 100

    kernel = Celerite.RealTerm(0.5, 1.0) + Celerite.SHOTerm(0.1, 2.0, -0.5)
    gp = Celerite.CeleriteGP(kernel)

    K = Celerite.get_matrix(gp, x)
    for n in 1:N
        K[n, n] = K[n, n] + yerr[n]^2
    end

    # Compute using Celerite
#    Celerite.compute!(gp, x, yerr)
    Celerite.compute_ldlt!(gp, x, yerr)
#    ll = Celerite.log_likelihood(gp, y)
    ll = Celerite.log_likelihood_ldlt(gp, y)

    # Compute directly
    ll0 = -0.5*(sum(y .* (K \ y)) + logdet(K) + N*log(2.0*pi))

    @test isapprox(ll, ll0)
end
