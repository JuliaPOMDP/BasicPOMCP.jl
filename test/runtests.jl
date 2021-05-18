using BasicPOMCP
using Test
using POMDPs
using POMDPModels
using NBInclude
using D3Trees
using Random
using POMDPSimulators
using POMDPModelTools
using POMDPTesting
using POMDPLinter: @requirements_info, @show_requirements, requirements_info
using ParticleFilters: n_particles, particles, particle, weights, weighted_particles, weight_sum, weight

import POMDPs:
	transition,
	observation,
    reward,
    discount,
	initialstate,
	updater,
	states,
	actions,
	observations

struct ConstObsPOMDP <: POMDP{Bool, Symbol, Bool} end
updater(problem::ConstObsPOMDP) = DiscreteUpdater(problem)
initialstate(::ConstObsPOMDP) = BoolDistribution(0.0)
transition(p::ConstObsPOMDP, s::Bool, a::Symbol) = BoolDistribution(0.0)
observation(p::ConstObsPOMDP, a::Symbol, sp::Bool) = BoolDistribution(1.0)
reward(p::ConstObsPOMDP, s::Bool, a::Symbol, sp::Bool) = 1.
discount(p::ConstObsPOMDP) = 0.9
states(p::ConstObsPOMDP) = (true, false)
actions(p::ConstObsPOMDP) = (:the_only_action,)
observations(p::ConstObsPOMDP) = (true, false)

@testset "POMDPTesting" begin
	pomdp = BabyPOMDP()
	test_solver(POMCPSolver(), BabyPOMDP())
end;

@testset "type stability" begin
	pomdp = BabyPOMDP()
	solver = POMCPSolver(rng = MersenneTwister(1))
	planner = solve(solver, pomdp)
	b = initialstate(pomdp)
    tree = BasicPOMCP.POMCPTree(pomdp, b, solver.tree_queries)
    node = BasicPOMCP.POMCPObsNode(tree, 1)

    r = @inferred BasicPOMCP.simulate(planner, rand(MersenneTwister(1), initialstate(pomdp)), node, 20)
end;

@testset "belief dependent actions" begin
	pomdp = ConstObsPOMDP()
	function POMDPs.actions(m::ConstObsPOMDP, b::LeafNodeBelief)
		@test currentobs(b) == true
        @test history(b)[end].o == true
        @test history(b)[end].a == :the_only_action
        @test mean(b) == 0.0
        @test mode(b) == 0.0
        @test only(support(b)) == false
        @test pdf(b, false) == 1.0
        @test pdf(b, true) == 0.0
        @test rand(b) == false
        @test n_particles(b) == 1
        @test only(particles(b)) == false
        @test only(weights(b)) == 1.0
        @test only(weighted_particles(b)) == (false => 1.0)
        @test weight_sum(b) == 1.0
        @test weight(b, 1) == 1.0 
        @test particle(b, 1) == false

        # old type name - this can be removed when upgrading versions
        @test b isa AOHistoryBelief
        return actions(m)
	end

	solver = POMCPSolver(rng = MersenneTwister(1))
	planner = solve(solver, pomdp)
	b = initialstate(pomdp)
    tree = BasicPOMCP.POMCPTree(pomdp, b, solver.tree_queries)
    node = BasicPOMCP.POMCPObsNode(tree, 1)

    @inferred BasicPOMCP.simulate(planner, rand(MersenneTwister(1), initialstate(pomdp)), node, 20)
end;

@testset "simulation" begin
	pomdp = BabyPOMDP()
	solver = POMCPSolver(rng = MersenneTwister(1))
	planner = solve(solver, pomdp)
    solver = POMCPSolver(max_time=0.1, tree_queries=typemax(Int), rng = MersenneTwister(1))
    planner = solve(solver, pomdp)
	b = initialstate(pomdp)

    a, info = action_info(planner, b)
    println("time below should be about 0.1 seconds")
    etime = @elapsed a, info = action_info(planner, b)
    @show etime
    @test etime < 0.2
    @show info[:search_time_us]

    sim = HistoryRecorder(max_steps=10)
    simulate(sim, pomdp, planner, updater(pomdp))
end;

@testset "d3t" begin
	pomdp = BabyPOMDP()
    solver = POMCPSolver(max_time=0.1, tree_queries=typemax(Int), rng = MersenneTwister(1))
    planner = solve(solver, pomdp)
	b = initialstate(pomdp)
    a, info = action_info(planner, b, tree_in_info=true)

    d3t = D3Tree(info[:tree], title="test tree")
    # inchrome(d3t)
    show(stdout, MIME("text/plain"), d3t)


    solver = POMCPSolver(max_time=0.1, tree_queries=typemax(Int), rng=MersenneTwister(1), tree_in_info=true)
    planner = solve(solver, pomdp)
    a, info = action_info(planner, b)

    d3t = D3Tree(info[:tree], title="test tree (tree_in_info solver option)")
end;

@testset "Minimal_Example" begin
    @nbinclude(joinpath(dirname(@__FILE__), "..", "notebooks", "Minimal_Example.ipynb"))
end;

@testset "consistency" begin
    # test consistency when rng is specified
    pomdp = BabyPOMDP()
    solver = POMCPSolver(rng = MersenneTwister(1))
    planner = solve(solver, pomdp)
    hist1 = simulate(HistoryRecorder(max_steps=1000, rng=MersenneTwister(3)), pomdp, planner)

    solver = POMCPSolver(rng = MersenneTwister(1))
    planner = solve(solver, pomdp)
    hist2 = simulate(HistoryRecorder(max_steps=1000, rng=MersenneTwister(3)), pomdp, planner)

    @test discounted_reward(hist1) == discounted_reward(hist2)
end;

@testset "requires" begin
    # REQUIREMENTS
    solver = POMCPSolver()
    pomdp = TigerPOMDP()

    println("============== @requirements_info with only solver:")
    requirements_info(solver)
    println("============== @requirements_info with solver and pomdp:")
    requirements_info(solver, pomdp)
    @show_requirements POMDPs.solve(solver, pomdp)
end;

@testset "errors" begin
    struct TerminalPOMDP <: POMDP{Int, Int, Int} end
    POMDPs.isterminal(::TerminalPOMDP, s) = true
    POMDPs.actions(::TerminalPOMDP) = [1,2,3]

    solver = POMCPSolver()
    planner = solve(solver, TerminalPOMDP())
    @test_throws AllSamplesTerminal action(planner, Deterministic(1))
    let ex = nothing
        try
            action(planner, Deterministic(1))
        catch ex
        end

        @test sprint(showerror, ex) == """
        Planner failed to choose an action because all states sampled from the belief were terminal.

        To see the belief, catch this exception as ex and see ex.belief.

        To specify an action for this case, use the default_action solver parameter.
        """
    end
end
