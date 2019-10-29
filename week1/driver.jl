include("gemm.jl")

using UnicodePlots
function plot_data(ns, data)
  scaling = 1e-9 .* (2 * ns.^3)
  init = true


  minv = floatmax()
  maxv = floatmin()
  for (s, v) = pairs(data)
    if s === :ref
      continue
    end
    minv = min(minv, minimum(scaling ./ v.min))
    maxv = max(maxv, maximum(scaling ./ v.min))
  end

  plt = lineplot(ns, scaling ./ data.ref.min, xlim = (ns[1], ns[end]),
                 ylim = (minv, maxv), title="max GFLOPS", xlabel="n",
                 ylabel="time", name = "ref")

  for (s, v) = pairs(data)
    if s == :ref
      continue
    end
    lineplot!(plt, ns, scaling ./ v.min, name = string(s))
  end
  plt
end


using BenchmarkTools

function run(ns, nsamples = 10)

  ref = (min=zeros(length(ns)), avg = zeros(length(ns)))
  ijp = (min=zeros(length(ns)), avg = zeros(length(ns)))
  ipj = (min=zeros(length(ns)), avg = zeros(length(ns)))
  pij = (min=zeros(length(ns)), avg = zeros(length(ns)))
  pji = (min=zeros(length(ns)), avg = zeros(length(ns)))
  jpi = (min=zeros(length(ns)), avg = zeros(length(ns)))
  jip = (min=zeros(length(ns)), avg = zeros(length(ns)))

  for (i, n) = enumerate(ns)
    @show i,n
    m = k = n
    C = rand(m, n)
    A = rand(m, k)
    B = rand(k, n)
    Cref = C + A * B

    bref = @benchmark $C += $A * $B samples=nsamples
    ref.min[i] = minimum(bref.times)
    ref.avg[i] = sum(bref.times) / length(bref.times)

    bijp = @benchmark mygemm_ijp($C, $A, $B) samples=nsamples
    ijp.min[i] = minimum(bijp.times)
    ijp.avg[i] = sum(bijp.times) / length(bijp.times)

    bipj = @benchmark mygemm_ipj($C, $A, $B) samples=nsamples
    ipj.min[i] = minimum(bipj.times)
    ipj.avg[i] = sum(bipj.times) / length(bipj.times)

    bpij = @benchmark mygemm_pij($C, $A, $B) samples=nsamples
    pij.min[i] = minimum(bpij.times)
    pij.avg[i] = sum(bpij.times) / length(bpij.times)

    bpji = @benchmark mygemm_pji($C, $A, $B) samples=nsamples
    pji.min[i] = minimum(bpji.times)
    pji.avg[i] = sum(bpji.times) / length(bpji.times)

    bjpi = @benchmark mygemm_jpi($C, $A, $B) samples=nsamples
    jpi.min[i] = minimum(bjpi.times)
    jpi.avg[i] = sum(bjpi.times) / length(bjpi.times)

    bjip = @benchmark mygemm_jip($C, $A, $B) samples=nsamples
    jip.min[i] = minimum(bjip.times)
    jip.avg[i] = sum(bjip.times) / length(bjip.times)
  end

  data = (ref = ref, ijp = ijp, ipj = ipj, pij = pij, pji = pji, jpi = jpi,
          jip = jip)

  return data
end

ns = 50:50:500
# data = run(ns)
plt = plot_data(ns, data)
