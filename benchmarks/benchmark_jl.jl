using Laplacians
using BenchmarkTools
using Printf

function main()
    if length(ARGS) < 1
        println(stderr, "Usage: julia benchmark_jl.jl <N> [samples=5]")
        exit(1)
    end

    n = parse(Int, ARGS[1])
    samples = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5

    # Generate random connected graph of size N with average degree 6
    a = rand_regular(n, 6)

    # Warmup
    f1_w = approxchol_lap(a)
    f2_w = approxchol_lap2(a)

    t_ac1 = @belapsed approxchol_lap($a) samples=samples seconds=5.0
    t_ac2 = @belapsed approxchol_lap2($a) samples=samples seconds=5.0

    @printf("julia\t%d\t%.3f\t%.3f\n", n, t_ac1 * 1000.0, t_ac2 * 1000.0)
end

main()
