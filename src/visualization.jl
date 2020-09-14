function D3Trees.D3Tree(p::POMCPPlanner; title="POMCP Tree", kwargs...)
    @warn("""
         D3Tree(planner::POMCPPlanner) is deprecated and may be removed in the future. Instead, please use

             a, info = action_info(planner, b)
             D3Tree(info[:tree])

         Or, you can get this info from a POMDPSimulators History

             info = first(ainfo_hist(hist))
             D3Tree(info[:tree])
         """)
    if p._tree == nothing
        error("POMCPPlanner has not constructed a tree yet, run `action(planner, belief)` first to construct the tree.")
    end
    return D3Tree(p._tree; title=title, kwargs...)
end

function D3Trees.D3Tree(t::POMCPTree; title="POMCP Tree", kwargs...)
    lenb = length(t.total_n)
    lenba = length(t.n)
    len = lenb + lenba
    children = Vector{Vector{Int}}(undef, len)
    text = Vector{String}(undef, len)
    tt = fill("", len)
    link_style = fill("", len)
    style = fill("", len)
    ba_children = [Set{Int}() for i in 1:lenba]
    for (ha_o, c) in t.o_lookup
        ha, o = ha_o
        push!(ba_children[ha], c)
    end
    min_V = minimum(t.v)
    max_V = maximum(t.v)
    for b in 1:lenb
        children[b] = t.children[b] .+ lenb
        text[b] = @sprintf("""
                           o: %s
                           N: %-10d""",
                           b==1 ? "<root>" : node_tag(t.o_labels[b]),
                           t.total_n[b]
                          )
        tt[b] = """
                o: $(b==1 ? "<root>" : node_tag(t.o_labels[b]))
                N: $(t.total_n[b])
                $(length(t.children[b])) children
                """
        link_width = max(1.0, 20.0*sqrt(t.total_n[b]/t.total_n[1]))
        link_style[b] = "stroke-width:$link_width"
    end
    for ba in 1:lenba
        children[ba+lenb] = collect(ba_children[ba])
        text[ba+lenb] = @sprintf("""
                                 a: %s
                                 N: %-7d\nV: %-10.3g""",
                                 node_tag(t.a_labels[ba]), t.n[ba], t.v[ba])
        tt[ba+lenb] = """
                      a: $(tooltip_tag(t.a_labels[ba]))
                      N: $(t.n[ba])
                      V: $(t.v[ba])
                      $(length(ba_children[ba])) children
                      """
        link_width = max(1.0, 20.0*sqrt(t.n[ba]/t.total_n[1]))
        link_style[ba+lenb] = "stroke-width:$link_width"
        rel_V = (t.v[ba]-min_V)/(max_V-min_V)
        if isnan(rel_V)
            color = colorant"gray"
        else
            color = weighted_color_mean(rel_V, colorant"green", colorant"red")
        end
        style[ba+lenb] = "stroke:#$(hex(color))"
    end
    return D3Tree(children;
                  text=text,
                  tooltip=tt,
                  style=style,
                  link_style=link_style,
                  title=title,
                  kwargs...
                 )

end

Base.show(io::IO, mime::MIME"text/html", t::POMCPTree) = show(io, mime, D3Tree(t))
Base.show(io::IO, mime::MIME"text/plain", t::POMCPTree) = show(io, mime, D3Tree(t))
