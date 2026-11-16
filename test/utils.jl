using Test
using Rimu
using KernelAbstractions
using QuantumNeuralStates
import QuantumNeuralStates: GPUGrowBuffer, ensure_capacity!

# CPU always; Metal only where a functional GPU exists
backends = Any[CPU()]
# Metal (Apple GPUs)
try
    @eval begin
        using Metal
        Metal.functional() && push!(backends, Metal.MetalBackend())
    end
catch
end

# CUDA (NVIDIA GPUs)
try
    @eval begin
        using CUDA
        CUDA.functional() && push!(backends, CUDA.CUDABackend())
    end
catch
end

@testset "GPUGrowBuffer" begin
    for backend in backends
        @testset "backend: $(nameof(typeof(backend)))" begin

            for fixed in ((), (3,), (2, 4))
                @testset "fixed = $fixed" begin
                    N   = length(fixed) + 1
                    buf = GPUGrowBuffer(backend, Float32, fixed, 10)

                    @testset "construction" begin
                        @test size(buf.data) == (fixed..., 10)
                        @test buf.capacity == 10
                        @test buf.fixed == fixed
                    end

                    @testset "within capacity: no reallocation" begin
                        d0 = buf.data
                        v  = ensure_capacity!(buf, 7)
                        @test size(v) == (fixed..., 7)
                        @test ndims(v) == N
                        @test buf.data === d0
                        @test buf.capacity == 10

                        v .= 1f0
                        @test sum(Array(v)) == prod(fixed) * 7
                    end

                    @testset "beyond capacity: grows by 1.2" begin
                        d0 = buf.data
                        v  = ensure_capacity!(buf, 25)
                        @test size(v) == (fixed..., 25)
                        @test buf.data !== d0
                        @test buf.capacity == ceil(Int, 25 * 1.2)        # 30
                        @test size(buf.data) == (fixed..., 30)

                        v .= 2f0
                        @test sum(Array(v)) == 2 * prod(fixed) * 25
                    end

                    @testset "smaller request keeps the allocation" begin
                        d1 = buf.data
                        v  = ensure_capacity!(buf, 5)
                        @test size(v) == (fixed..., 5)
                        @test buf.data === d1
                        @test buf.capacity == 30
                    end

                    @testset "exact capacity does not grow" begin
                        d1 = buf.data
                        v  = ensure_capacity!(buf, 30)
                        @test size(v) == (fixed..., 30)
                        @test buf.data === d1
                    end
                end
            end

            @testset "vector constructor" begin
                buf = GPUGrowBuffer(backend, Float32, 8)
                @test buf.fixed == ()
                @test size(buf.data) == (8,)
                @test ndims(ensure_capacity!(buf, 3)) == 1
            end

            @testset "isbits addresses" begin
                addr = BoseFS{missing}(1, 0, 2)
                abuf = GPUGrowBuffer(backend, typeof(addr), 4)

                va = ensure_capacity!(abuf, 3)
                fill!(va, addr)
                @test size(va) == (3,)
                @test all(==(addr), Array(va))

                a2 = BoseFS{missing}(0, 3, 0)
                va = ensure_capacity!(abuf, 9)                           # grow
                fill!(va, a2)
                @test size(va) == (9,)
                @test abuf.capacity == ceil(Int, 9 * 1.2)                # 11
                @test all(==(a2), Array(va))
            end
        end
    end
end
