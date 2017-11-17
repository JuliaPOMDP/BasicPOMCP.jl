using BasicPOMCP
using Base.Test
using POMDPs
using POMDPModels
using POMDPToolbox
using NBInclude
using D3Trees

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

solver = POMCPSolver(max_time=0.1, tree_queries=typemax(Int), rng = MersenneTwister(1))
planner = solve(solver, pomdp)
action(planner, initial_state_distribution(pomdp))

d3t = D3Tree(planner, title="test")
# inchrome(d3t)

nbinclude(joinpath(dirname(@__FILE__), "..", "notebooks", "Minimal_Example.ipynb"))

d3t = D3Tree(planner, title="test")
# inchrome(d3t)

# test consistency when rng is specified
pomdp = BabyPOMDP()
solver = POMCPSolver(rng = MersenneTwister(1))
planner = solve(solver, pomdp)
hist1 = simulate(HistoryRecorder(max_steps=1000, rng=MersenneTwister(3)), pomdp, planner)

solver = POMCPSolver(rng = MersenneTwister(1))
planner = solve(solver, pomdp)
hist2 = simulate(HistoryRecorder(max_steps=1000, rng=MersenneTwister(3)), pomdp, planner)

@test discounted_reward(hist1) == discounted_reward(hist2)

# REQUIREMENTS
solver = POMCPSolver()
pomdp = TigerPOMDP()

println("============== @requirements_info with only solver:")
@requirements_info solver
println("============== @requirements_info with solver and pomdp:")
@requirements_info solver pomdp
