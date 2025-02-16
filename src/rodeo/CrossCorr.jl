"""
    CrossCorr

Computes cross-correlation functions and time differences between signals.
This module is part of the `rodeo` Python package.

William F. Jenkins, Ph.D.
Scripps Institution of Oceanography
University of California San Diego
February 2025
"""

module CrossCorr

using DSP
using LinearAlgebra
using ProgressBars
using Statistics


"""
    cross_corr(sig1::Vector{Float64}, sig2::Vector{Float64})::Float64

Compute the maximum value of the cross-correlation function between two signals.
"""
function cross_corr(sig1::Vector{Float64}, sig2::Vector{Float64})
    x1 = (sig1 .- mean(sig1)) ./ std(sig1)
    x2 = (sig2 .- mean(sig2)) ./ std(sig2)
    norm_factor = sqrt(sum(x1.^2) * sum(x2.^2))
    ccf = xcorr(x1, x2) ./ norm_factor
    return maximum(ccf)
end

"""
    corr_matrix(data::Matrix{Float64})::Matrix{Float64}

Compute the matrix containing maximum values of the cross-correlation functions
between pairs of signals.
"""
function corr_matrix(data::Matrix{Float64})::Matrix{Float64}
    rows = size(data, 1)
    corr = Matrix{Float64}(undef, rows, rows)

    Threads.@threads for i in ProgressBar(1:rows)
        sig1 = data[i, :]
        for j in i:rows
            sig2 = data[j, :]
            ccf = cross_corr(sig1, sig2)
            corr[i, j] = ccf
            corr[j, i] = ccf
        end
    end
    return corr
end

"""
    dt_matrix(times::Vector{Float64})::Matrix{Float64}

Compute the matrix containing time differences between pairs of time points.
"""
function dt_matrix(times::Vector{Float64})::Matrix{Float64}
    rows = length(times)
    dt = Matrix{Float64}(undef, rows, rows)

    Threads.@threads for i in ProgressBar(1:rows)
        for j in i:rows
            dt[i, j] = times[j] - times[i]
            dt[j, i] = -dt[i, j]
        end
    end
    return dt
end

export corr_matrix, dt_matrix

end