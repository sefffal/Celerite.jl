using Optim
using PyPlot
#import Celerite
include("../src/Celerite.jl")

srand(42)

# The input coordinates must be sorted
t = sort(cat(1, 3.8 * rand(57), 5.5 + 4.5 * rand(68)))
yerr = 0.08 + (0.22-0.08)*rand(length(t))
y = 0.2*(t-5.0) + sin.(3.0*t + 0.1*(t-5.0).^2) + yerr .* randn(length(t))

true_t = linspace(0, 10, 1000)
true_y = 0.2*(true_t-5) + sin.(3*true_t + 0.1*(true_t-5).^2)

PyPlot.clf()
PyPlot.plot(true_t, true_y, "k", lw=1.5, alpha=0.3)
PyPlot.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
PyPlot.xlabel("x")
PyPlot.ylabel("y")
PyPlot.xlim(0, 10)
PyPlot.ylim(-2.5, 2.5);

Q = 1.0 / sqrt(2.0)
w0 = 3.0
S0 = var(y) / (w0 * Q)
kernel = Celerite.SHOTerm(log(S0), log(Q), log(w0))

# A periodic component
Q = 1.0
w0 = 3.0
S0 = var(y) / (w0 * Q)
kernel = kernel + Celerite.SHOTerm(log(S0), log(Q), log(w0))

Celerite.TermSum((Celerite.SHOTerm(-0.7432145976901582,-0.34657359027997275,1.0986122886681096),Celerite.SHOTerm(-1.089788187970131,0.0,1.0986122886681096)))

gp = Celerite.CeleriteGP(kernel)
Celerite.compute!(gp, t, yerr)
Celerite.log_likelihood(gp, y)

#mu, variance = Celerite.predict_full_ldlt(gp, y, true_t, return_var=true)
#mu, variance = Celerite.predict_full(gp, y, true_t, return_var=true)
mu, variance = Celerite.predict(gp, y, true_t, return_var=true)
#mu  = Celerite.predict_full(gp, y, true_t, return_cov=false, return_var=false)
#mu  = Celerite.predict!(gp, gp.x, y, true_t)
sigma = sqrt.(variance)

PyPlot.plot(true_t, true_y, "k", lw=1.5, alpha=0.3,label="Simulated data")
PyPlot.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
PyPlot.plot(true_t, mu, "g",label="Initial kernel")
PyPlot.fill_between(true_t, mu+sigma, mu-sigma, color="g", alpha=0.3)
PyPlot.xlabel("x")
PyPlot.ylabel("y")
PyPlot.xlim(0, 10)
PyPlot.ylim(-2.5, 2.5);
read(STDIN,Char)

PyPlot.clf()
vector = Celerite.get_parameter_vector(gp.kernel)
mask = ones(Bool, length(vector))
mask[2] = false  # Don't fit for the first Q
function nll(params)
    vector[mask] = params
    Celerite.set_parameter_vector!(gp.kernel, vector)
    Celerite.compute!(gp, t, yerr)
    return -Celerite.log_likelihood(gp, y)
end;

result = Optim.optimize(nll, vector[mask], Optim.LBFGS())
result

vector[mask] = Optim.minimizer(result)
vector

Celerite.set_parameter_vector!(gp.kernel, vector)

#mu, variance = Celerite.predict(gp, y, true_t, return_var=true)
#mu, variance = Celerite.predict_full_ldlt(gp, y, true_t, return_var=true)
#mu = Celerite.predict(gp, y, true_t, return_cov = false, return_var=false)
#mu = Celerite.predict!(gp, gp.x, y, true_t)
mu, variance = Celerite.predict(gp, y, true_t, return_var=true)
sigma = sqrt.(variance)

PyPlot.plot(true_t, true_y, "k", lw=1.5, alpha=0.3)
PyPlot.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
PyPlot.plot(true_t, mu, "r",label="Optimized kernel")
PyPlot.fill_between(true_t, mu+sigma, mu-sigma, color="g", alpha=0.3)
PyPlot.xlabel("x")
PyPlot.ylabel("y")
PyPlot.xlim(0, 10)
PyPlot.ylim(-2.5, 2.5);
PyPlot.legend(loc="upper left")

read(STDIN,Char)
PyPlot.clf()

omega = exp.(linspace(log(0.1), log(20), 5000))
psd = Celerite.get_psd(gp.kernel, omega)

for term in gp.kernel.terms
    PyPlot.plot(omega, Celerite.get_psd(term, omega), "--", color="orange")
end
PyPlot.plot(omega, psd, color="orange")

PyPlot.yscale("log")
PyPlot.xscale("log")
PyPlot.xlim(omega[1], omega[end])
PyPlot.xlabel("omega")
PyPlot.ylabel("S(omega)");
PyPlot.plot(2*pi/2.0*[1.,1.],[1e-8,1e2])
