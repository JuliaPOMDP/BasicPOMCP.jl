function POMDPs.requirements_info(solver::AbstractPOMCPSolver, problem::POMDP)
    if @implemented initialstate_distribution(::typeof(problem))
        return requirements_info(solver, problem, initialstate_distribution(problem))
    else
        println("""
        Since POMCP is an online solver, most of the computation occurs in `action(planner, state)`. In order to view the requirements for this function, please, supply a state as the third argument to `requirements_info`, e.g.

            @requirements_info $(typeof(solver))() $(typeof(problem))() $(statetype(typeof(problem)))()

            """)
    end
end

function POMDPs.requirements_info(solver::AbstractPOMCPSolver, problem::POMDP, b)
    policy = solve(solver, problem)
    requirements_info(policy, b)
end

function POMDPs.requirements_info(policy::POMCPPlanner, b)
    @show_requirements action(policy, b)

    problem = policy.problem
    rng = MersenneTwister(1)
    if @implemented(rand(::typeof(rng), ::typeof(b))) &&
        @implemented(actions(::typeof(problem), ::typeof(b)))
        s = rand(rng, b)
        a = first(actions(problem, b))
        if @implemented generate_sor(::typeof(policy.problem), ::typeof(s), ::typeof(a), ::typeof(rng))
            sp, o, r = generate_sor(policy.problem, s, a, rng)

            if !isequal(deepcopy(o), o)
                @warn("""
                     isequal(deepcopy(o), o) returned false. Is isequal() defined correctly?

                     For POMCP to work correctly, you must define isequal(::$(typeof(o)), ::$(typeof(o))) (see https://docs.julialang.org/en/stable/stdlib/collections/#Associative-Collections-1, https://github.com/andrewcooke/AutoHashEquals.jl#background, also consider using StaticArrays). This warning was thrown because isequal($(deepcopy(o)), $o) returned false.

                     Note: isequal() should also be defined correctly for actions, but no warning will be issued.
                     """)
            end
            if hash(deepcopy(o)) != hash(o)
                @warn("""
                     hash(deepcopy(o)) was not equal to hash(o). Is hash() defined correctly?

                     For MCTS to work correctly, you must define hash(::$(typeof(o)), ::UInt) (see https://docs.julialang.org/en/stable/stdlib/collections/#Associative-Collections-1, https://github.com/andrewcooke/AutoHashEquals.jl#background, also consider using StaticArrays). This warning was thrown because hash($(deepcopy(o))) != hash($o).

                     Note: hash() should also be defined correctly for actions, but no warning will be issued.
                     """)
            end
        end
    end
end

@POMDP_require action(p::POMCPPlanner, b) begin
    tree = POMCPTree(p.problem, p.solver.tree_queries)
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
    @req generate_sor(::P, ::S, ::A, ::typeof(p.rng))
    @req isequal(::O, ::O)
    @req hash(::O)
    # from insert_obs_node!
    @req n_actions(::P)
    @req actions(::P, ::typeof(hnode))
    AS = typeof(actions(p.problem, hnode))
    @subreq estimate_value(p.solved_estimator, p.problem, s, hnode, steps)
    @req discount(::P)
end

@POMDP_require estimate_value(f::Function, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int) begin
    @req f(::typeof(pomdp), ::typeof(start_state), ::typeof(h), ::typeof(steps))
end
