using BasicPOMCP
using POMDPModels
using POMDPs

m = BabyPOMDP()
solver = POMCPSolver()
planner = solve(solver, m)
b = initialstate(m)
@show action(planner, b)
