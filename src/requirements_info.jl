function POMDPLinter.requirements_info(solver::AbstractPOMCPSolver, problem::POMDP)
    println("""
    Since POMCP is an online solver, most of the computation occurs in `action(planner, state)`. In order to view the requirements for this function, please, supply an initial beleif to `requirements_info`, e.g.

            @requirements_info $(typeof(solver))() $(typeof(problem))() initialstate(pomdp)

        """)
end

function POMDPLinter.requirements_info(solver::AbstractPOMCPSolver, problem::POMDP, b)
    policy = solve(solver, problem)
    requirements_info(policy, b)
end

POMDPs.requirements_info(policy::POMCPPlanner, b) = @show_requirements action(policy, b)

@POMDP_require action(p::POMCPPlanner, b) begin
    tree = POMCPTree(p.problem, b, p.solver.tree_queries)
    @subreq search(p, b, tree)
end

@POMDP_require search(p::POMCPPlanner, b, t::POMCPTree) begin
    P = typeof(p.problem)
    @req rand(::typeof(p.rng), ::typeof(b))
    s = rand(p.rng, b)
    @req isterminal(::P, ::statetype(P))
    @subreq simulate(p, s, POMCPObsNode(t, 1), p.solver.max_depth)
end

@POMDP_require simulate(p::POMCPPlanner, s, hnode::POMCPObsNode, steps::Int) begin
    P = typeof(p.problem)
    S = statetype(P)
    A = actiontype(P)
    O = obstype(P)
    @req gen(::P, ::S, ::A, ::typeof(p.rng))
    @req isequal(::O, ::O)
    @req hash(::O)
    # from insert_obs_node!
    @req actions(::P)
    AS = typeof(actions(p.problem))
    @req length(::AS)
    @subreq estimate_value(p.solved_estimator, p.problem, s, hnode, steps)
    @req discount(::P)
end

@POMDP_require estimate_value(f::Function, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int) begin
    @req f(::typeof(pomdp), ::typeof(start_state), ::typeof(h), ::typeof(steps))
end
