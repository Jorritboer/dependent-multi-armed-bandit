import cvxpy as cp

x = cp.Variable()
y = cp.Variable()

p1max = cp.Parameter()
# p2max = cp.Parameter()
# p3max = cp.Parameter()

eq1 = 0.5 * x + y

constraints = [
    x >= 0,
    y >= 0,
    eq1 <= p1max,
]

prob1 = cp.Problem(cp.Maximize(x), constraints)

p1max.value = 0.25

xmax = prob1.solve(solver=cp.SCS, requires_grad=True)
print(xmax)

prob1.backward()
print(p1max.gradient)
