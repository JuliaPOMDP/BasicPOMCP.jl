abstract type NoDecision <: Exception end
Base.show(io::IO, nd::NoDecision) = print(io, """
    Planner failed to choose an action because the following exception was thrown:
    $nd

    To specify an action for this case, use the default_action solver parameter.
    """)

struct AllSamplesTerminal <: NoDecision
    belief
end
Base.show(io::IO, ast::AllSamplesTerminal) = print(io, """
    Planner failed to choose an action because all states sampled from the belief were terminal.

    To see the belief, catch this exception as ex and see ex.belief.
    
    To specify an action for this case, use the default_action solver parameter.
    """)


# Default action stuff was moved to MCTS
# This section can be deleted once a version of MCTS with default actions is tagged
if !isdefined(MCTS, :default_action)
    struct ExceptionRethrow end

    default_action(::ExceptionRethrow, pomdp, belief, ex) = rethrow(ex)
    default_action(f::Function, pomdp, belief, ex) = f(belief, ex)
    default_action(p::POMDPs.Policy, pomdp, belief, ex) = action(p, belief)
    default_action(s::POMDPs.Solver, pomdp, belief, ex) = action(solve(s, pomdp), belief)
    default_action(a, pomdp, belief, ex) = a

    """
        ReportWhenUsed(a)

    When the planner fails, returns action `a`, but also prints the exception.
    """
    struct ReportWhenUsed{T}
        a::T
    end

    function default_action(r::ReportWhenUsed, pomdp, belief, ex)
        showerror(STDERR, ex)
        a = default_action(r.a, pomdp, belief, ex)
        warn("Using default action $a")
        return a
    end
end
