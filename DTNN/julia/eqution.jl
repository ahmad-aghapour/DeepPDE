using Flux
using Random

abstract type Equation end

"""
Base class for defining PDE related function.

# Arguments
- `eqn_config::Dict`: dictionary containing PDE configuration parameters

# Attributes
- `dim::Int`: dimensionality of the problem
- `total_time::Float64`: total time horizon
- `num_time_interval::Int`: number of time steps
- `delta_t::Float64`: time step size
- `sqrt_delta_t::Float64`: square root of time step size
- `y_init`: initial value of the function
"""
mutable struct BaseEquation <: Equation
    dim::Int
    total_time::Float64
    num_time_interval::Int
    delta_t::Float64
    sqrt_delta_t::Float64
    y_init

    function BaseEquation(eqn_config::Dict{Symbol, Any})
        new_eqn = new()
        new_eqn.dim = eqn_config[:dim]
        new_eqn.total_time = eqn_config[:total_time]
        new_eqn.num_time_interval = eqn_config[:num_time_interval]
        new_eqn.delta_t = new_eqn.total_time / new_eqn.num_time_interval
        new_eqn.sqrt_delta_t = sqrt(new_eqn.delta_t)
        new_eqn.y_init = nothing
        return new_eqn
    end
end

function sample end

function r_u end

function h_z end

function terminal end


mutable struct PricingDefaultRisk <: Equation
    dim::Int
    total_time::Float64
    num_time_interval::Int
    delta_t::Float64
    sqrt_delta_t::Float64
    y_init::Any
    x_init::Array{Float64}
    sigma::Float64
    rate::Float64
    delta::Float64
    gammah::Float64
    gammal::Float64
    mu_bar::Float64
    K::Int
    vh::Float64
    vl::Float64
    slope::Float64

    function PricingDefaultRisk(eqn_config)
        new_pdr = new()
        new_pdr.dim = eqn_config["dim"]
        new_pdr.total_time = eqn_config["total_time"]
        new_pdr.num_time_interval = eqn_config["num_time_interval"]
        new_pdr.delta_t = new_pdr.total_time / new_pdr.num_time_interval
        new_pdr.sqrt_delta_t = sqrt(new_pdr.delta_t)
        new_pdr.y_init = nothing

        new_pdr.x_init = ones(1,new_pdr.dim) .* 100
        new_pdr.sigma = 0.2
        new_pdr.rate = 0.02
        new_pdr.delta = 2.0 / 3
        new_pdr.gammah = 0.2
        new_pdr.gammal = 0.02
        new_pdr.mu_bar = 0.02
        new_pdr.K = 100
        new_pdr.vh = 50.0
        new_pdr.vl = 70.0
        new_pdr.slope = (new_pdr.gammah - new_pdr.gammal) / (new_pdr.vh - new_pdr.vl)
        return new_pdr
    end
end

function sample(pdr::PricingDefaultRisk, num_sample::Int)
    dw_sample = randn(num_sample, pdr.dim, pdr.num_time_interval) .* pdr.sqrt_delta_t
    x_sample = zeros(num_sample, pdr.dim, pdr.num_time_interval + 1)
    x_sample[:, :, 1] .=  ones(num_sample,pdr.dim) .* pdr.x_init
    for i in 1:pdr.num_time_interval
        x_sample[:, :, i + 1] .= (1 + pdr.mu_bar * pdr.delta_t) .* x_sample[:, :, i] + (pdr.sigma .* x_sample[:, :, i] .* dw_sample[:, :, i])
    end
    return dw_sample, x_sample
end

function r_u(pdr::PricingDefaultRisk, t::Float64, x, y, z)
    piecewise_linear = relu.(relu.(y .- pdr.vh) .* pdr.slope + pdr.gammah - pdr.gammal) .+ pdr.gammal
    return ((1 - pdr.delta) .* piecewise_linear .+ pdr.rate)
end

function h_z(pdr::PricingDefaultRisk, t::Float64, x, y, z)
    return zeros(size(x, 1), 1)
end

function terminal(pdr::PricingDefaultRisk, t::Float64, x)
    return relu.(minimum(x, dims=2))
end

function terminal_for_sample(pdr::PricingDefaultRisk, x)
    return relu.(minimum(x, dims=3))
end




mutable struct BlackScholesBarenblatt <: Equation
    dim::Int
    total_time::Float64
    num_time_interval::Int
    delta_t::Float64
    sqrt_delta_t::Float64
    y_init::Any
    x_init::Matrix{Float64}
    sigma::Float64
    rate::Float64
    mu_bar::Float64

    function BlackScholesBarenblatt(eqn_config)
        new_pdr = new()
        new_pdr.dim = eqn_config["dim"]
        new_pdr.total_time = eqn_config["total_time"]
        new_pdr.num_time_interval = eqn_config["num_time_interval"]
        new_pdr.delta_t = new_pdr.total_time / new_pdr.num_time_interval
        new_pdr.sqrt_delta_t = sqrt(new_pdr.delta_t)
        new_pdr.y_init = nothing


        new_pdr.x_init = reshape([1.0 / (1.0 + i % 2) for i in 1:new_pdr.dim],(1,new_pdr.dim))
        new_pdr.sigma = 0.4
        new_pdr.rate = 0.05
        new_pdr.mu_bar = 0.0
        return new_pdr
    end
end

function sample(equation::BlackScholesBarenblatt, num_sample::Int)
    dw_sample = randn(num_sample, equation.dim, equation.num_time_interval) .* equation.sqrt_delta_t
    x_sample = zeros(num_sample, equation.dim, equation.num_time_interval + 1)
    x_sample[:, :, 1] .= ones(num_sample,equation.dim) .* equation.x_init
    for i in 1:equation.num_time_interval
        x_sample[:, :, i + 1] .= (1 + equation.mu_bar * equation.delta_t) .* x_sample[:, :, i] .+ 
            (equation.sigma .* x_sample[:, :, i] .* dw_sample[:, :, i])
    end
    return dw_sample, x_sample
end

function r_u(equation::BlackScholesBarenblatt, t::Float64, x, y, z)
    return ones(size(x, 1)) .* equation.rate
end

function h_z(equation::BlackScholesBarenblatt, t::Float64, x, y, z)
    return -1 * sum(z, dims=2) .* equation.rate / equation.sigma
end

function terminal(equation::BlackScholesBarenblatt, t::Float64, x)
    return sum(x.^2, dims=2)
end

function terminal_for_sample(equation::BlackScholesBarenblatt, x)
    return sum(x.^2, dims=3)
end



mutable struct PricingDiffRate <: Equation
    dim::Int
    total_time::Float64
    num_time_interval::Int
    delta_t::Float64
    sqrt_delta_t::Float64
    y_init::Any
    x_init::Matrix{Float64}
    sigma::Float64
    mu_bar::Float64
    rl::Float64
    rb::Float64
    alpha::Float64

    function PricingDiffRate(eqn_config)
        obj = new()
        obj.dim = eqn_config["dim"]
        obj.total_time = eqn_config["total_time"]
        obj.num_time_interval = eqn_config["num_time_interval"]
        obj.delta_t = obj.total_time / obj.num_time_interval
        obj.sqrt_delta_t = sqrt(obj.delta_t)
        obj.y_init = nothing
        obj.x_init = ones(1,obj.dim) .* 100
        obj.sigma = 0.2
        obj.mu_bar = 0.06
        obj.rl = 0.04
        obj.rb = 0.06
        obj.alpha = 1.0 / obj.dim
        return obj
    end
end

function sample(equation::PricingDiffRate, num_sample::Int)
    dw_sample = randn(num_sample, equation.dim, equation.num_time_interval) .* equation.sqrt_delta_t
    x_sample = zeros(num_sample, equation.dim, equation.num_time_interval + 1)
    x_sample[:, :, 1] = ones(num_sample,equation.dim) .* equation.x_init
    factor = exp((equation.mu_bar - (equation.sigma^2) / 2) * equation.delta_t)
    for i in 1:equation.num_time_interval
        x_sample[:, :, i + 1] .= (factor .* exp(equation.sigma * dw_sample[:, :, i])) .* x_sample[:, :, i]
    end
    return dw_sample, x_sample
end

function r_u(equation::PricingDiffRate, t::Float64, x, y, z)
    temp = sum(z, dims=2) ./ equation.sigma .- y
    return Flux.where(temp .> 0, equation.rb, equation.rl)
end

function h_z(equation::PricingDiffRate, t::Float64, x, y, z)
    temp = sum(z, dims=2) ./ equation.sigma .- y
    return Flux.where(temp .> 0, (equation.mu_bar - equation.rb) .* sum(z, dims=2) ./ equation.sigma, (equation.mu_bar - equation.rl) .* sum(z, dims=2) ./ equation.sigma)
end

function terminal(equation::PricingDiffRate, t::Float64, x)
    temp = maximum(x, dims=2)
    return max.(temp .- 120, 0)
end

function terminal_for_sample(equation::PricingDiffRate, x)
    temp = maximum(x, dims=3)
    return max.(temp .- 120, 0)
end
