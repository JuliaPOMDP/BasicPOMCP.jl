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

test_solver(POMCPSolver(), BabyPOMDP())

pomdp = BabyPOMDP()
solver = POMCPSolver(rng = MersenneTwister(1))
planner = solve(solver, pomdp)

tree = BasicPOMCP.POMCPTree(pomdp, solver.tree_queries)
node = BasicPOMCP.POMCPObsNode(tree, 1)

r = @inferred BasicPOMCP.simulate(planner, initialstate(pomdp, MersenneTwister(1)), node, 20)

sim = HistoryRecorder(max_steps=10)
simulate(sim, pomdp, planner, updater(pomdp))

solver = POMCPSolver(max_time=0.1, tree_queries=typemax(Int), rng = MersenneTwister(1))
planner = solve(solver, pomdp)
a, info = action_info(planner, initialstate_distribution(pomdp))
a, info = action_info(planner, initialstate_distribution(pomdp))
println("time below should be about 0.1 seconds")
etime = @elapsed a, info = action_info(planner, initialstate_distribution(pomdp))
@show etime
@test etime < 0.2 
@show info[:search_time_us]

solver = POMCPSolver(max_time=0.1, tree_queries=typemax(Int), rng = MersenneTwister(1))
planner = solve(solver, pomdp)
a, info = action_info(planner, initialstate_distribution(pomdp), tree_in_info=true)

#d3t = D3Tree(planner, title="test")
d3t = D3Tree(info[:tree], title="test tree")
# inchrome(d3t)
show(stdout, MIME("text/plain"), d3t)


solver = POMCPSolver(max_time=0.1, tree_queries=typemax(Int), rng=MersenneTwister(1), tree_in_info=true)
planner = solve(solver, pomdp)
a, info = action_info(planner, initialstate_distribution(pomdp))

d3t = D3Tree(info[:tree], title="test tree (tree_in_info solver option)")

@nbinclude(joinpath(dirname(@__FILE__), "..", "notebooks", "Minimal_Example.ipynb"))

#d3t = D3Tree(planner, title="test")
# inchrome(d3t)

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
end

@testset "requires" begin
    # REQUIREMENTS
    solver = POMCPSolver()
    pomdp = TigerPOMDP()

    println("============== @requirements_info with only solver:")
    @requirements_info solver
    println("============== @requirements_info with solver and pomdp:")
    @requirements_info solver pomdp
end
