using Test
using EngineGUI
using EngineGUI.EngineT

@testset "EngineGUI.jl Tests" begin
    @testset "Module Loading" begin
        @test isdefined(EngineGUI, :launch_gui)
        @test isdefined(EngineT, :Engine)
    end

    @testset "Engine Creation" begin
        # Test engine initialization with a simple model
        @test_throws Exception Engine("invalid-model-name")
    end

    @testset "Basic Functionality" begin
        # Test that launch_gui function exists and accepts parameters
        @test hasmethod(launch_gui, (String,))
        @test hasmethod(launch_gui, ())
    end
end
