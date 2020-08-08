x = 7

sqrt(x) + rand()

# Unicode characters can be easily entered by typing the corresponding
#  LaTeX code (like \alpha), and then hitting the Tab key.

α = 2
α
"A string with various Unicode characters (α, β and ϕ)."

function square(x)
    return x^2
end

square(6)

using Pkg
Pkg.add("Gadfly")

using Gadfly
plot([x -> sin(x) / x], 0, 50)
