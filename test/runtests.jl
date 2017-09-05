using BasicPOMCP
using Base.Test
using POMDPs
using POMDPModels
using POMDPToolbox
using NBInclude

test_solver(POMCPSolver(), BabyPOMDP())

pomdp = BabyPOMDP()
solver = POMCPSolver(rng = MersenneTwister(1))
planner = solve(solver, pomdp)

tree = BasicPOMCP.POMCPTree(pomdp, solver.tree_queries)
node = BasicPOMCP.POMCPObsNode(tree, 1)

r = @inferred BasicPOMCP.simulate(planner, initial_state(pomdp, MersenneTwister(1)), node, 20)

sim = HistoryRecorder(max_steps=10)
simulate(sim, pomdp, planner, updater(pomdp))

solver = POMCPSolver(max_time=0.1, tree_queries=typemax(Int), rng = MersenneTwister(1))
planner = solve(solver, pomdp)
action(planner, initial_state_distribution(pomdp))
println("time below should be about 0.1 seconds")
@time action(planner, initial_state_distribution(pomdp))

nbinclude(joinpath(dirname(@__FILE__), "..", "notebooks", "Minimal_Example.ipynb"))
