
module BasicPOMCP

#=
Current constraints:
- action space discrete
- action space same for all states, histories
- no built-in support for history-dependent rollouts (this could be added though)
- initial n and initial v are 0
=#

using POMDPs
using Parameters
using ParticleFilters
using BeliefUpdaters
using POMDPPolicies
using POMDPSimulators
using CPUTime
using Colors
using Random
using Printf

import POMDPs: action, solve, updater, requirements_info
import POMDPModelTools: action_info

using MCTS
import MCTS: convert_estimator, estimate_value, node_tag, tooltip_tag, default_action

using D3Trees

export
    POMCPSolver,
    POMCPPlanner,

    action,
    solve,
    updater,

    NoDecision,
    AllSamplesTerminal,
    ExceptionRethrow,
    ReportWhenUsed,
    default_action,

    BeliefNode,
    AbstractPOMCPSolver,

    PORollout,
    FORollout,
    RolloutEstimator,
    FOValue,

    D3Tree,
    node_tag,
    tooltip_tag

abstract type AbstractPOMCPSolver <: Solver end

"""
    POMCPSolver(#=keyword arguments=#)

Partially Observable Monte Carlo Planning Solver.

## Keyword Arguments

- `max_depth::Int`
    Rollouts and tree expension will stop when this depth is reached.
    default: `20`

- `c::Float64`
    UCB exploration constant - specifies how much the solver should explore.
    default: `1.0`

- `tree_queries::Int`
    Number of iterations during each action() call.
    default: `1000`

- `max_time::Float64`
    Maximum time for planning in each action() call.
    default: `Inf`

- `tree_in_info::Bool`
    If `true`, returns the tree in the info dict when action_info is called.
    default: `false`

- `estimate_value::Any`
    Function, object, or number used to estimate the value at the leaf nodes.
    default: `RolloutEstimator(RandomSolver(rng))`
    - If this is a function `f`, `f(pomdp, s, h::BeliefNode, steps)` will be called to estimate the value.
    - If this is an object `o`, `estimate_value(o, pomdp, s, h::BeliefNode, steps)` will be called.
    - If this is a number, the value will be set to that number
    Note: In many cases, the simplest way to estimate the value is to do a rollout on the fully observable MDP with a policy that is a function of the state. To do this, use `FORollout(policy)`.

- `default_action::Any`
    Function, action, or Policy used to determine the action if POMCP fails with exception `ex`.
    default: `ExceptionRethrow()`
    - If this is a Function `f`, `f(pomdp, belief, ex)` will be called.
    - If this is a Policy `p`, `action(p, belief)` will be called.
    - If it is an object `a`, `default_action(a, pomdp, belief, ex)` will be called, and if this method is not implemented, `a` will be returned directly.

- `rng::AbstractRNG`
    Random number generator.
    default: `Random.GLOBAL_RNG`
"""
@with_kw mutable struct POMCPSolver <: AbstractPOMCPSolver
    max_depth::Int          = 20
    c::Float64              = 1.0
    tree_queries::Int       = 1000
    max_time::Float64       = Inf
    tree_in_info::Bool      = false
    default_action::Any     = ExceptionRethrow()
    rng::AbstractRNG        = Random.GLOBAL_RNG
    estimate_value::Any     = RolloutEstimator(RandomSolver(rng))
end

struct POMCPTree{A,O}
    # for each observation-terminated history
    total_n::Vector{Int}                 # total number of visits for an observation node
    children::Vector{Vector{Int}}        # indices of each of the children
    o_labels::Vector{O}                  # actual observation corresponding to this observation node

    o_lookup::Dict{Tuple{Int, O}, Int}   # mapping from (action node index, observation) to an observation node index

    # for each action-terminated history
    n::Vector{Int}                       # number of visits for an action node
    v::Vector{Float64}                   # value estimate for an action node
    a_labels::Vector{A}                  # actual action corresponding to this action node
end

function POMCPTree(pomdp::POMDP, sz::Int=1000)
    acts = collect(actions(pomdp))
    A = actiontype(pomdp)
    O = obstype(pomdp)
    sz = min(100_000, sz)
    return POMCPTree{A,O}(sizehint!(Int[0], sz),
                          sizehint!(Vector{Int}[collect(1:length(acts))], sz),
                          sizehint!(Array{O}(undef, 1), sz),

                          sizehint!(Dict{Tuple{Int,O},Int}(), sz),

                          sizehint!(zeros(Int, length(acts)), sz),
                          sizehint!(zeros(Float64, length(acts)), sz),
                          sizehint!(acts, sz)
                         )
end    

function insert_obs_node!(t::POMCPTree, pomdp::POMDP, ha::Int, o)
    push!(t.total_n, 0)
    push!(t.children, sizehint!(Int[], n_actions(pomdp)))
    push!(t.o_labels, o)
    hao = length(t.total_n)
    t.o_lookup[(ha, o)] = hao
    for a in actions(pomdp)
        n = insert_action_node!(t, hao, a)
        push!(t.children[hao], n)
    end
    return hao
end

function insert_action_node!(t::POMCPTree, h::Int, a)
    push!(t.n, 0)
    push!(t.v, 0.0)
    push!(t.a_labels, a)
    return length(t.n)
end

abstract type BeliefNode <: AbstractStateNode end

struct POMCPObsNode{A,O} <: BeliefNode
    tree::POMCPTree{A,O}
    node::Int
end

mutable struct POMCPPlanner{P, SE, RNG} <: Policy
    solver::POMCPSolver
    problem::P
    solved_estimator::SE
    rng::RNG
    _best_node_mem::Vector{Int}
    _tree::Union{Nothing, Any}
end

function POMCPPlanner(solver::POMCPSolver, pomdp::POMDP)
    se = convert_estimator(solver.estimate_value, solver, pomdp)
    return POMCPPlanner(solver, pomdp, se, solver.rng, Int[], nothing)
end

Random.seed!(p::POMCPPlanner, seed) = Random.seed!(p.rng, seed)


function updater(p::POMCPPlanner)
    P = typeof(p.problem)
    S = statetype(P)
    A = actiontype(P)
    O = obstype(P)
    if !@implemented ParticleFilters.obs_weight(::P, ::S, ::A, ::S, ::O)
        return UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
    end
    return SIRParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
end

# TODO (maybe): implement this for history-dependent policies
#=
immutable AOHistory
    tree::POMCPTree
    tail::Int
end

length
getindex
=#

include("solver.jl")

include("exceptions.jl")
include("rollout.jl")
include("visualization.jl")
include("requirements_info.jl")

end # module
