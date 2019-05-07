function action_info(p::POMCPPlanner, b; tree_in_info=false)
    local a::actiontype(p.problem)
    info = Dict{Symbol, Any}()
    try
        tree = POMCPTree(p.problem, b, p.solver.tree_queries)
        a = search(p, b, tree, info)
        p._tree = tree
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
    catch ex
        # Note: this might not be type stable, but it shouldn't matter too much here
        a = convert(actiontype(p.problem), default_action(p.solver.default_action, p.problem, b, ex))
        info[:exception] = ex
    end
    return a, info
end

action(p::POMCPPlanner, b) = first(action_info(p, b))

function search(p::POMCPPlanner, b, t::POMCPTree, info::Dict)
    all_terminal = true
    nquery = 0
    start_us = CPUtime_us()
    for i in 1:p.solver.tree_queries
        nquery += 1
        if CPUtime_us() - start_us >= 1e6*p.solver.max_time
            break
        end
        s = rand(p.rng, b)
        if !POMDPs.isterminal(p.problem, s)
            simulate(p, s, POMCPObsNode(t, 1), p.solver.max_depth)
            all_terminal = false
        end
    end
    info[:search_time_us] = CPUtime_us() - start_us
    info[:tree_queries] = nquery

    if all_terminal
        throw(AllSamplesTerminal(b))
    end

    h = 1
    best_node = first(t.children[h])
    best_v = t.v[best_node]
    @assert !isnan(best_v)
    for node in t.children[h][2:end]
        if t.v[node] >= best_v
            best_v = t.v[node]
            best_node = node
        end
    end

    return t.a_labels[best_node]
end

solve(solver::POMCPSolver, pomdp::POMDP) = POMCPPlanner(solver, pomdp)

function simulate(p::POMCPPlanner, s, hnode::POMCPObsNode, steps::Int)
    if steps == 0 || isterminal(p.problem, s)
        return 0.0
    end
    
    t = hnode.tree
    h = hnode.node

    ltn = log(t.total_n[h])
    best_nodes = empty!(p._best_node_mem)
    best_criterion_val = -Inf
    for node in t.children[h]
        n = t.n[node]
        if n == 0 && ltn <= 0.0
            criterion_value = t.v[node]
        elseif n == 0 && t.v[node] == -Inf
            criterion_value = Inf
        else
            criterion_value = t.v[node] + p.solver.c*sqrt(ltn/n)
        end
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            empty!(best_nodes)
            push!(best_nodes, node)
        elseif criterion_value == best_criterion_val
            push!(best_nodes, node)
        end
    end
    ha = rand(p.rng, best_nodes)
    a = t.a_labels[ha]

    sp, o, r = generate_sor(p.problem, s, a, p.rng)

    hao = get(t.o_lookup, (ha, o), 0)
    if hao == 0
        hao = insert_obs_node!(t, p.problem, ha, o)
        v = estimate_value(p.solved_estimator,
                           p.problem,
                           sp,
                           POMCPObsNode(t, hao),
                           steps-1)
        R = r + discount(p.problem)*v
    else
        R = r + discount(p.problem)*simulate(p, sp, POMCPObsNode(t, hao), steps-1)
    end

    t.total_n[h] += 1
    t.n[ha] += 1
    t.v[ha] += (R-t.v[ha])/t.n[ha]

    return R
end
