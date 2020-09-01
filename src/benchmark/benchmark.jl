include("utils.jl")

function fairevaluate(
    classifiers::Array{<:MLJBase.Model,1}, X, y;
    measure = nothing,
    grp::Symbol = :class,
    priv_grps=nothing,
    random_seed::Int=12345,
    n_folds=6
)
    if priv_grps!=nothing
        @assert(all(
            priv_grp in levels(X[!, grp]) for priv_grp in priv_grps
            for classifier in classifiers
        ))
    end
    @assert(all(
        target_scitype(classifier) <: AbstractVector{<:Finite}
        for classifier in classifiers
    ))

    n = length(classifiers)

    measure = measure == nothing ? DisparateImpact(grp, priv_grps) : measure
    results = zeros(n, n_folds)
    cv = CV(nfolds = n_folds, shuffle=true, rng=random_seed)
    for i = 1:n
        operation = classifiers[i] isa Probabilistic ? MLJBase.predict_mode : MLJBase.predict
        result = evaluate(
            classifiers[i],
            X,
            y,
            resampling = cv,
            measure = measure,
            operation = operation,
        )
        results[i, :] = result.per_fold[1]
    end
    pvalues, tstats = zeros(n, n), zeros(n, n)
    for i = 1:n
        for j = 1:n
            ttestResult = OneSampleTTest(results[i, :], results[j, :])
            tstats[i, j] = ttestResult.t
            pvalues[i, j] = pvalue(ttestResult)
        end
    end
    return pvalues, tstats
end
