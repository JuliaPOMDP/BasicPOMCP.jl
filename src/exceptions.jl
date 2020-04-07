abstract type NoDecision <: Exception end
Base.showerror(io::IO, nd::NoDecision) = print(io, """
    Planner failed to choose an action because the following exception was thrown:
    $nd

    To specify an action for this case, use the default_action solver parameter.
    """)

struct AllSamplesTerminal <: NoDecision
    belief
end
Base.showerror(io::IO, ast::AllSamplesTerminal) = print(io, """
    Planner failed to choose an action because all states sampled from the belief were terminal.

    To see the belief, catch this exception as ex and see ex.belief.

    To specify an action for this case, use the default_action solver parameter.
    """)
