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
using POMDPToolbox
using ParticleFilters

import POMDPs: action, solve, updater

using MCTS

export
    POMCPSolver,
    POMCPPlanner,

    NoDecision,
    AllSamplesTerminal,
    ExceptionRethrow,
    default_action


@with_kw type POMCPSolver <: Solver
    max_depth::Int          = 20
    c::Float64              = 1.0
    tree_queries::Int       = 1000
    estimate_value::Any     = RolloutEstimator(RandomSolver())
    default_action::Any     = ExceptionRethrow()
    rng::AbstractRNG        = Base.GLOBAL_RNG
end

type POMCPPlanner{P, SE, RNG} <: Policy
    solver::POMCPSolver
    problem::P
    solved_estimator::SE
    rng::RNG
end

immutable POMCPTree{A,O}
    # for each observation-terminated history
    total_n::Vector{Int}
    children::Vector{Vector{Int}}
    o_labels::Vector{O}

    o_lookup::Dict{Tuple{Int, O}, Int}

    # for each action-terminated history
    n::Vector{Int}
    v::Vector{Float64}
    a_labels::Vector{A}
end
function POMCPTree(pomdp::POMDP, sz::Int=1000)
    acts = collect(iterator(actions(pomdp)))
    A = action_type(pomdp)
    O = obs_type(pomdp)
    return POMCPTree{A,O}(sizehint!(Int[0], sz),
                          sizehint!(Vector{Int}[collect(1:length(acts))], sz),
                          sizehint!(Array(O,1), sz),

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
    for a in iterator(actions(pomdp))
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

immutable POMCPObsNode{A,O}
    tree::POMCPTree{A,O}
    node::Int
end

function updater(p::POMCPPlanner)
    P = typeof(p.problem)
    S = state_type(P)
    A = action_type(P)
    O = obs_type(P)
    if !@implemented ParticleFilters.obs_weight(::P, ::S, ::A, ::S, ::O)
        warn("""
             The default belief updater for a POMCPSolver is the `SIRParticleFilter` from ParticleFilters.jl. However this requires `ParticleFilters.obs_weight(::$P, ::$S, ::$A, ::$S, ::$O)` You can still use the POMCPSolver without this, but simulation with this updater will probably fail. See the documentation for `ParticleFilters.obs_weight` for more details.
            """)
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

end # module
