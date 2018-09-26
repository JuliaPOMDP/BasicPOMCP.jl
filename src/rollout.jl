struct PORollout
    solver::Union{POMDPs.Solver,POMDPs.Policy,Function}
    updater::POMDPs.Updater
end

struct SolvedPORollout{P<:POMDPs.Policy,U<:POMDPs.Updater,RNG<:AbstractRNG}
    policy::P
    updater::U
    rng::RNG
end

struct FORollout # fully observable rollout
    solver::Union{POMDPs.Solver,POMDPs.Policy}
end

struct SolvedFORollout{P<:POMDPs.Policy,RNG<:AbstractRNG}
    policy::P
    rng::RNG
end

struct FOValue
    solver::Union{POMDPs.Solver, POMDPs.Policy}
end

struct SolvedFOValue{P<:POMDPs.Policy}
    policy::P
end

"""
    estimate_value(estimator, problem::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int)

Return an initial unbiased estimate of the value at belief node h.

By default this runs a rollout simulation
"""
function estimate_value end
estimate_value(f::Function, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int) = f(pomdp, start_state, h, steps)
estimate_value(n::Number, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int) = convert(Float64, n)

function estimate_value(estimator::Union{SolvedPORollout,SolvedFORollout}, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int)
    rollout(estimator, pomdp, start_state, h, steps)
end

@POMDP_require estimate_value(estimator::Union{SolvedPORollout,SolvedFORollout}, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int) begin
    @subreq rollout(estimator, pomdp, start_state, h, steps)
end

function estimate_value(estimator::SolvedFOValue, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int)
    POMDPs.value(estimator.policy, start_state)
end


function convert_estimator(ev::RolloutEstimator, solver, pomdp)
    policy = MCTS.convert_to_policy(ev.solver, pomdp)
    SolvedPORollout(policy, updater(policy), solver.rng)
end

function convert_estimator(ev::PORollout, solver, pomdp)
    policy = MCTS.convert_to_policy(ev.solver, pomdp)
    SolvedPORollout(policy, ev.updater, solver.rng)
end

function convert_estimator(est::FORollout, solver, pomdp)
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    SolvedFORollout(policy, solver.rng)
end

function convert_estimator(est::FOValue, solver::AbstractPOMCPSolver, pomdp::POMDPs.POMDP)
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    SolvedFOValue(policy)
end


"""
Perform a rollout simulation to estimate the value.
"""
function rollout(est::SolvedPORollout, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int)
    b = extract_belief(est.updater, h)
    sim = RolloutSimulator(est.rng,
                           steps)
    return POMDPs.simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end

@POMDP_require rollout(est::SolvedPORollout, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int) begin
    @req extract_belief(::typeof(est.updater), ::typeof(h))
    b = extract_belief(est.updater, h)
    sim = RolloutSimulator(est.rng,
                           steps)
    @subreq POMDPs.simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end


function rollout(est::SolvedFORollout, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int)
    sim = RolloutSimulator(est.rng,
                                        steps)
    return POMDPs.simulate(sim, pomdp, est.policy, start_state)
end

@POMDP_require rollout(est::SolvedFORollout, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int) begin
    sim = RolloutSimulator(est.rng,
                                        steps)
    @subreq POMDPs.simulate(sim, pomdp, est.policy, start_state)
end



"""
    extract_belief(rollout_updater::POMDPs.Updater, node::BeliefNode)

Return a belief compatible with the `rollout_updater` from the belief in `node`.

When a rollout simulation is started, this function is used to create the initial belief (compatible with `rollout_updater`) based on the appropriate `BeliefNode` at the edge of the tree. By overriding this, a belief can be constructed based on the entire tree or entire observation-action history.
"""
function extract_belief end

# some defaults are provided
extract_belief(::NothingUpdater, node::BeliefNode) = nothing

function extract_belief(::PreviousObservationUpdater, node::BeliefNode)
    if node.node==1 && !isdefined(node.tree.o_labels, node.node)
        missing
    else
        node.tree.o_labels[node.node]
    end
end
