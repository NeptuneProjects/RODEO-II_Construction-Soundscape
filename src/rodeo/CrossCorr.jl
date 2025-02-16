module CrossCorr

using DSP
using LinearAlgebra
using ProgressBars
using Statistics


# Function to compute max cross-correlation between two signals
function cross_corr(sig1::Vector{Float64}, sig2::Vector{Float64})
    x1 = (sig1 .- mean(sig1)) ./ std(sig1)
    x2 = (sig2 .- mean(sig2)) ./ std(sig2)
    norm_factor = sqrt(sum(x1.^2) * sum(x2.^2))
    ccf = xcorr(x1, x2) ./ norm_factor
    return maximum(ccf)
end

# Function to compute cross-correlation matrix
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

# Compute the time difference between signals
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

# Export primary function for Python access
export corr_matrix, dt_matrix

end # module