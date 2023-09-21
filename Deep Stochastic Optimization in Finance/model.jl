using Distributions
using BSON
using CUDA
using Flux
using Flux: @functor, chunk
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using Logging: with_logger
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random
"""
    samplePrice(
        phi::Float64 = 0.2,
        nT::Int = 22,
        s0::Float64 = 100.0,
        v0::Float64 = 0.04,
        k::Float64 = 0.9,
        theta::Float64 = 0.04,
        sigma::Float64 = 0.2,
        lambda::Float64 = 0.0,
        mu::Float64 = 0.0
    ) -> (Array{Float64,1}, Float64)

Simulate the Heston model dynamics to calculate the price of an asset.

- `phi`: Correlation between the two Brownian motions.
- `nT`: Number of time steps.
- `s0`: Initial stock price.
- `v0`: Initial variance.
- `k` : Heston model parameter
- `theta` : Heston model parameter
- `sigma` : Heston model Volatility
- `lambda` : Heston model parameter
- `mu` : Heston model mean


Returns an array of asset prices and the final price.

.. math::
    dS_t = μS_t dt + √v_t S_t dW_t^1
    dv_t = k(θ - v_t)dt + σ√v_t dW_t^2

Where `dW_t^1` and `dW_t^2` are correlated Brownian motions with correlation `phi`.
"""
function samplePrice(;
        phi::Float64 = 0.2,
        nT::Int = 22,
        s0::Float64 = 100.0,
        v0::Float64 = 0.04,
        k::Float64 = 0.9,
        theta::Float64 = 0.04,
        sigma::Float64 = 0.2,
        lambda::Float64 = 0.0,
        mu::Float64 = 0.0
    )

    dt = 1.0 / (12.0 * nT)
    covarianceMatrix = [dt 0; 0 dt]
    distribution = MvNormal([0.0, 0.0], covarianceMatrix)
    samples = transpose(rand(distribution, nT + 2))

    assetPrices = zeros(nT + 1)
    assetPrices[1] = s0

    for i in 2:nT + 1
        assetPrices[i] = assetPrices[i-1] * (1 + mu * dt + sqrt(v0) * samples[i, 1])
        v0 += (k * (theta - v0) - lambda * v0) * dt + sigma * v0 * (phi * samples[i, 1] + sqrt(1 - phi^2) * samples[i, 2])
    end

    return assetPrices, assetPrices[end]
end

"""
    generateData(
        nsample::Int = 100000,
        ...
    ) -> (Array{Array{Float64,1},1}, Array{Float64,1})

Generate `nsample` stock data samples based on the Heston model dynamics.


Returns an array of asset price sequences and an array of final prices for each sequence.
"""
function generateData(;
        nsample::Int = 100000,
        phi::Float64 = 0.2,
        nT::Int = 22,
        s0::Float64 = 100.0,
        v0::Float64 = 0.04,
        k::Float64 = 0.9,
        theta::Float64 = 0.04,
        sigma::Float64 = 0.2,
        lambda::Float64 = 0.0,
        mu::Float64 = 0.0
    )

    assetSamples = []
    finalPrices = Float64[]

    for _ in 1:nsample
        t, s = samplePrice(;phi = phi, nT = nT, s0 = s0, v0 = v0, k = k, theta = theta, sigma = sigma, lambda = lambda, mu = mu)
        assetSample, finalPrice = samplePrice(;phi = phi, nT = nT, s0 = s0, v0 = v0, k = k, theta = theta, sigma = sigma, lambda = lambda, mu = mu)
        push!(assetSamples, Float64.(assetSample))
        push!(finalPrices, finalPrice)
    end

    return assetSamples, Float64.(finalPrices)
end


using Flux

"""
    prepareIndicesAndValues(a::Array{Float64, 1})::Array{Array{Float64, 1}, 1}

Prepare a new array of index-value pairs based on input array `a`.

Each pair is represented as: [index, value].

Returns a new array of such index-value pairs.
"""
function prepareIndicesAndValues(a::Array{Float64, 1})
    return [[i-1, a[i]] for i in 1:length(a)]
end


"""
computeReturns(a::Array{Float64, 1})::Array{Float64, 1}

Calculate the returns for the given price sequence `a`.

.. math::
    r_t = \frac{a_t - a_{t-1}}{a_{t-1}}

Where r_t is the return at time t.
Returns an array of computed returns.
"""

function computeReturns(a::Array{Float64, 1})
    return [(a[i] - a[i-1]) / a[i-1] for i in 2:length(a)]
end


"""
    getData(
        nsample::Int = 100000,
        batchSize::Int = 512,
        phi::Float64 = 0.2,
        nT::Int = 22,
        s0::Float64 = 100.0,
        v0::Float64 = 0.04,
        k::Float64 = 0.9,
        theta::Float64 = 0.04,
        sigma::Float64 = 0.2,
        lambda::Float64 = 0.0,
        mu::Float64 = 0.0
    )::DataLoader

Generate data samples for training.

- `nsample`: Number of samples.
... [similar explanations for the rest of the parameters]

The data is generated using the Heston dynamics (not provided in the given code).

Returns a DataLoader containing batches of index-value pairs, returns, and y-values.
"""
function getData(;
        nsample::Int = 100000,
        batchSize::Int = 512,
        phi::Float64 = 0.2,
        nT::Int = 22,
        s0::Float64 = 100.0,
        v0::Float64 = 0.04,
        k::Float64 = 0.9,
        theta::Float64 = 0.04,
        sigma::Float64 = 0.2,
        lambda::Float64 = 0.0,
        mu::Float64 = 0.0
    )

    xtrain, ytrain = generateData(;nsample=nsample, phi=phi, nT=nT, s0=s0, v0=v0, k=k, theta=theta, sigma=sigma, lambda=lambda, mu=mu)
    indicesValues = [hcat(prepareIndicesAndValues(xtrain[i])...) for i in 1:nsample]
    returns = [computeReturns(xtrain[i]) for i in 1:nsample]

    return Flux.Data.DataLoader((indicesValues, returns, ytrain), batchsize=batchSize, shuffle=true)
end





struct Action
    layer::Chain
end

@functor Action

"""
    Action(inputDim::Int, latentDim::Int, hiddenDim::Int)

Construct an `Action` struct initialized with dense layers based on dimensions provided.
"""
function Action(
        inputDim::Int,
        latentDim::Int,
        hiddenDim::Int
    )

    return Action(Chain(
        Dense(inputDim, hiddenDim, relu),
        Dense(hiddenDim, hiddenDim, relu),
        Dense(hiddenDim, 1)
    ))
end

function (action::Action)(x)
    return action.layer(x)
end

"""
    wealth(action::Action, x1, x2, x, r, nT, batchSize)

Compute the wealth for a given action using the input values.

Returns the calculated wealth.
"""
function wealth(
        action::Action,
        x1,
        x2,
        x,
        r,
        nT::Int,
        batchSize::Int
    )

    h = action(x1)
    h = reshape(h, nT, batchSize)
    out = x

    for i in 2:size(h, 1)
        out = (1 + r) .* out .+ h[i-1, :] .* (x2[i-1, :] .- r)
    end

    return out
end


"""
    modelLoss(action::Action, x1, x2, x, r, y, K, nT, batchSize)

Compute the model's loss using mean squared error.

Returns the computed loss.
"""
function modelLoss(
        action::Action,
        x1,
        x2,
        x,
        r,
        y,
        K,
        nT::Int,
        batchSize::Int
    )

    wealthValue = wealth(action, x1, x2, x, r, nT, batchSize)
    Flux.Losses.mse(wealthValue, max.(y .- K, 0))
end

"""
    Args

A mutable structure to encapsulate the various arguments used in the training process.
"""
@with_kw mutable struct Args
    η::Float64 = 1e-3  # learning rate
    nsample::Int = 1000000
    batchSize::Int = 512
    phi::Float64 = 0.2
    nT::Int = 22
    s0::Float64 = 100.0
    v0::Float64 = 0.04
    k::Float64 = 0.9
    x0::Array{Float64,1} = [1.0]
    theta::Float64 = 0.04
    sigma::Float64 = 0.2
    lambda::Float64 = 0.0
    mu::Float64 = 0.0
    r::Float64 = 0.0
    Ki::Int = 100
    epochs::Int = 6
    seed::Int = 0
    useGpu::Bool = true
    inputDim::Int = 2
    latentDim::Int = 2
    hiddenDim::Int = 20
    verboseFreq::Int = 10
    tbLogger::Bool = false
end




"""
    train(; kwargs...)

Train a model using given hyperparameters and data. The function initializes and
trains an `Action` struct using the MNIST dataset and ADAM optimizer.

# Parameters
- `kwargs`: Hyperparameters required for the training procedure.

# Returns
- The function does not explicitly return values. It updates the model parameters and
  displays training information.
...
"""
function train(
        ; kws...
    )

    # Load hyperparameters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU configuration
    device = (args.useGpu && CUDA.has_cuda()) ? gpu : cpu
    infoMessage = (device == gpu) ? "Training on GPU" : "Training on CPU"
    @info infoMessage

    # Load data
    loader = getData(;
        nsample=args.nsample,
        batchSize=args.batchSize,
        phi=args.phi,
        nT=args.nT,
        s0=args.s0,
        v0=args.v0,
        k=args.k,
        theta=args.theta,
        sigma=args.sigma,
        lambda=args.lambda,
        mu=args.mu
    )

    actionModel = Action(args.inputDim, args.latentDim, args.hiddenDim) |> device

    # ADAM optimizer
    opt = ADAM(args.η)
    x = args.x0 |> device

    # Parameters
    ps = Flux.params(actionModel.layer, x)

    # Training
    trainSteps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch in 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))

        for (x1, x2, y) in loader
            x1 = hcat(x1...)
            x1 = reshape(x1, 2, args.nT + 1, Int(round(size(x1)[2] / (args.nT + 1)))) |> device

            x2 = hcat(x2...) |> device
            y = y |> device


            loss = modelLoss(
                actionModel, x1, x2, x, args.r, y, args.Ki |> device, size(x1)[2], size(x1)[3]
            )

            grad = gradient(ps) do
                modelLoss(actionModel, x1, x2, x, args.r, y,  args.Ki |> device, size(x1)[2], size(x1)[3])
            end

            update!(opt, ps, grad)

            # Progress meter
            next!(progress; showvalues=[(:loss, loss), (:x, x)])

            # Logging with TensorBoard
            if args.tbLogger && trainSteps % args.verboseFreq == 0
                with_logger(tbLogger) do
                    @info "train" loss=loss
                    @info "x" x=x
                end
            end

            trainSteps += 1
        end
    end
end
