using Flux

"""
Neural network model for time dimension.
"""
struct TimeNet <: Flux.Chain
    layer1::Flux.Dense
    tanh1::Flux.Tanh
    layer2::Flux.Dense
    tanh2::Flux.Tanh
    layer3::Flux.Dense
    tanh3::Flux.Tanh
    layer4::Flux.Dense
    tanh4::Flux.Tanh
    layer5::Flux.Dense
    tanh5::Flux.Tanh
    layer6::Flux.Dense
end

function TimeNet(output_dim::Int)
    l1 = Dense(4, 100)
    t1 = Tanh()
    l2 = Dense(100, 150)
    t2 = Tanh()
    l3 = Dense(150, 200)
    t3 = Tanh()
    l4 = Dense(200, 300)
    t4 = Tanh()
    l5 = Dense(300, 200)
    t5 = Tanh()
    l6 = Dense(200, output_dim)
    TimeNet(l1, t1, l2, t2, l3, t3, l4, t4, l5, t5, l6)
end

Flux.@functor TimeNet

function (m::TimeNet)(x::AbstractArray)
    x = m.layer1(x)
    x = m.tanh1(x)
    x = m.layer2(x)
    x = m.tanh2(x)
    x = m.layer3(x)
    x = m.tanh3(x)
    x = m.layer4(x)
    x = m.tanh4(x)
    x = m.layer5(x)
    x = m.tanh5(x)
    x = m.layer6(x)
    return x
end

"""
A struct for defining a neural network with a single linear layer.
"""
struct Net1 <: Flux.Chain
    layer::Flux.Dense
    bn::Flux.BatchNorm
end

function Net1(input_dim::Int, output_dim::Int)
    l = Dense(input_dim, output_dim)
    b = BatchNorm(output_dim)
    Net1(l, b)
end

Flux.@functor Net1

function (m::Net1)(x::AbstractArray)
    x = m.layer(x)
    return x
end


using Flux
using Flux: glorot_uniform
using NNlib: relu, softmax

struct MAB
    fc_q::Dense
    fc_k::Dense
    fc_v::Dense
    fc_o::Dense
    ln0::LayerNorm
    ln1::LayerNorm
    dim_v::Int
    num_heads::Int
end

function MAB(dim_q, dim_k, dim_v, num_heads; ln=false)
    layers = (
        Dense(dim_q, dim_v),
        Dense(dim_k, dim_v),
        Dense(dim_k, dim_v),
        Dense(dim_v, dim_v),
    )
    if ln
        return MAB(layers..., LayerNorm(dim_v), LayerNorm(dim_v), dim_v, num_heads)
    else
        return MAB(layers..., LayerNorm(0), LayerNorm(0), dim_v, num_heads)
    end
end

function (m::MAB)(q, k)
    q = m.fc_q(q)
    k, v = m.fc_k(k), m.fc_v(k)

    dim_split = m.dim_v ÷ m.num_heads
    q_ = vcat(split(q, dim_split, dims=3)...)
    k_ = vcat(split(k, dim_split, dims=3)...)
    v_ = vcat(split(v, dim_split, dims=3)...)

    a = softmax(q_ * permutedims(k_) / sqrt(float(m.dim_v)), dims=2)
    o = hcat(split(q_ + a * v_, size(q, 1), dims=1)...)
    o = m.ln0.α != 0 ? m.ln0(o) : o
    o = o + relu(m.fc_o(o))
    o = m.ln1.α != 0 ? m.ln1(o) : o
    return o
end

struct SAB
    mab::MAB
end

SAB(dim_in, dim_out, num_heads; ln=false) = SAB(MAB(dim_in, dim_in, dim_out, num_heads, ln=ln))
(m::SAB)(x) = m.mab(x, x)

struct ISAB
    mab0::MAB
    mab1::MAB
    i::AbstractArray
end

function ISAB(dim_in, dim_out, num_heads, num_inds; ln=false)
    i = param(glorot_uniform(1, num_inds, dim_out))
    mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
    mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
    ISAB(mab0, mab1, i)
end

function (m::ISAB)(x)
    h = m.mab0(repeat(m.i, outer=(size(x, 1), 1, 1)), x)
    return m.mab1(x, h)
end

struct PMA
    mab::MAB
    s::AbstractArray
end

function PMA(dim, num_heads, num_seeds; ln=false)
    s = param(glorot_uniform(1, num_seeds, dim))
    mab = MAB(dim, dim, dim, num_heads, ln=ln)
    PMA(mab, s)
end

(m::PMA)(x) = m.mab(repeat(m.s, outer=(size(x, 1), 1, 1)), x)


using Flux

struct TimeNetForSet
    feature
    layer1
    layer2
    layer3
    layer4
    relu1
    relu2
    relu3
end

function TimeNetForSet(in_features::Int=1, out_features::Int=64)
    feature = Dense(in_features, out_features)
    layer1 = Dense(1, 10)
    layer2 = Dense(10, 10)
    layer3 = Dense(10, 10)
    layer4 = Dense(10, out_features)
    relu1 = relu
    relu2 = relu
    relu3 = relu
    TimeNetForSet(feature, layer1, layer2, layer3, layer4, relu1, relu2, relu3)
end

function (m::TimeNetForSet)(t, x)
    t = m.relu1(m.layer1(t))
    t = m.relu2(m.layer2(t))
    t = m.relu3(m.layer3(t))
    t = m.layer4(t)
    x = m.feature(x)
    x = exp.(t) .* x
    return x
end


struct DeepTimeSetTransformer
    feature_extractor
    time_layer1
    time_layer2
    time_layer3
    time_layer4
    time_layer5
    layer1
    layer2
    layer3
    layer4
    layer5
    regressor
end

function DeepTimeSetTransformer(input_dim::Int)
    feature_extractor = Chain(
        Dense(1, 64, relu),
        Dense(64, 64, relu),
        Dense(64, 64, relu),
        Dense(64, 64)
    )
    
    time_layer1 = TimeNetForSet(input_dim, 32)
    time_layer2 = TimeNetForSet(32, 32)
    time_layer3 = TimeNetForSet(32, 32)
    time_layer4 = TimeNetForSet(32, 32)
    time_layer5 = TimeNetForSet(32, 1)

    layer1 = Dense(input_dim, 32)
    layer2 = Dense(32, 32)
    layer3 = Dense(32, 32)
    layer4 = Dense(32, 32)
    layer5 = Dense(32, 32)

    regressor = Chain(PMA(32, 4, 1))

    DeepTimeSetTransformer(feature_extractor, time_layer1, time_layer2, time_layer3, time_layer4, time_layer5, layer1, layer2, layer3, layer4, layer5, regressor)
end

function (m::DeepTimeSetTransformer)(t, x)
    x = m.layer1(x)
    x = relu.(x)
    x = x .- mean(x, dims=2)
    x = m.layer2(x)

    x = relu.(x)
    x = x .- mean(x, dims=2)
    x = m.layer3(x)

    x = relu.(x)
    x = x .- mean(x, dims=2)
    x = m.layer4(x)
    
    x = relu.(x)
    x = x .- mean(x, dims=2)
    x = m.layer5(x)
    
    output = relu.(x)

    output = m.regressor(output)
    output = squeeze(output)

    output = m.time_layer2(t, output)
    output = relu.(output)
    output = m.time_layer3(t, output)
    output = relu.(output)
    output = m.time_layer4(t, output)
    output = relu.(output)
    output = m.time_layer5(t, output)

    return output
end


using Flux

mutable struct TimeDependentNetwork
    n_layer::Int
    layers::Vector
    time_layer::Vector
    batch_layer::Vector
    linear
end

function TimeDependentNetwork(indim::Int, layersize::Vector{Int}, outdim::Int)
    n_layer = length(layersize)
    layers = [Net1(indim, layersize[1])]
    time_layer = [TimeNet(indim + 3)] # for concatenated tensor
    batch_layer = [BatchNorm(indim), BatchNorm(layersize[1])]

    for i in 2:n_layer
        push!(layers, Net1(layersize[i-1], layersize[i]))
        push!(time_layer, TimeNet(layersize[i-1] + 3))
        push!(batch_layer, BatchNorm(layersize[i]))
    end

    linear = Dense(layersize[end], outdim)

    TimeDependentNetwork(n_layer, layers, time_layer, batch_layer, linear)
end

function (m::TimeDependentNetwork)(t, x)
    for i in 1:m.n_layer
        time = m.time_layer[i](vcat(t, t.^2, t.^3, exp.(t)))
        x = x .* (1 .+ time)
        x = relu.(m.layers[i](x))
    end
    return m.linear(x)
end

mutable struct TimeDependentNetworkMonteCarlo
    n_layer::Int
    layers::Vector
    time_layer::Vector
    batch_layer::Vector
    linear
    sigma::Float32
end

function TimeDependentNetworkMonteCarlo(indim::Int, layersize::Vector{Int}, outdim::Int, sigma::Float32)
    n_layer = length(layersize)
    layers = [Net1(indim, layersize[1])]
    time_layer = [TimeNet(indim + 3)] # for concatenated tensor
    batch_layer = [BatchNorm(indim), BatchNorm(layersize[1])]

    for i in 2:n_layer
        push!(layers, Net1(layersize[i-1], layersize[i]))
        push!(time_layer, TimeNet(layersize[i-1] + 3))
        push!(batch_layer, BatchNorm(layersize[i]))
    end

    linear = Dense(layersize[end], outdim)

    TimeDependentNetworkMonteCarlo(n_layer, layers, time_layer, batch_layer, linear, sigma)
end

function (m::TimeDependentNetworkMonteCarlo)(t, x, y)
    x_prim = x

    for i in 1:m.n_layer
        time = m.time_layer[i](vcat(t, t.^2, t.^3, exp.(t)))
        x = x .* (1 .+ time)
        x = relu.(m.layers[i](x))
    end

    time = m.time_layer[end](vcat(t, t.^2, t.^3, exp.(t)))
    return m.sigma * x_prim .* m.linear(x) + (1 .- time) .* y
end
