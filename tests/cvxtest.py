import cvxpy as cp

v1 = 0.5
v2 = 0.2

bandit_probs = [v1, v1 + v2, v2, 0.7 * v1 + v2]
# bandit_probs = [v1, v1 + v2, v2, 0.7 * v1 + v2]

v1 = cp.Variable()
v2 = cp.Variable()

p1max = cp.Parameter()
p1min = cp.Parameter()
p2max = cp.Parameter()
p2min = cp.Parameter()
p3max = cp.Parameter()
p3min = cp.Parameter()
p4max = cp.Parameter()
p4min = cp.Parameter()

eq1 = v1
eq2 = v1 + v2
eq3 = v2
eq4 = 0.7 * v1 + v2

constraints = [
    v1 >= 0,
    v1 <= 1,
    v2 >= 0,
    v2 <= 1,
    eq1 >= p1min,
    eq1 <= p1max,
    eq2 >= p2min,
    eq2 <= p2max,
    eq3 >= p3min,
    eq3 <= p3max,
    eq4 >= p4min,
    eq4 <= p4max,
]

prob1 = cp.Problem(cp.Maximize(eq1), constraints)
prob2 = cp.Problem(cp.Maximize(eq2), constraints)
prob3 = cp.Problem(cp.Maximize(eq3), constraints)
prob4 = cp.Problem(cp.Maximize(eq4), constraints)

print(v1.gradient)
print(v2.gradient)

# v1.gradient = 1
# v2.gradient = 1


def dep_max_est(estimators, bound):
    p1min.value = estimators[0] - bound[0]
    p1max.value = estimators[0] + bound[0]
    p2min.value = estimators[1] - bound[1]
    p2max.value = estimators[1] + bound[1]
    p3min.value = estimators[2] - bound[2]
    p3max.value = estimators[2] + bound[2]
    p4min.value = estimators[3] - bound[3]
    p4max.value = estimators[3] + bound[3]

    def print_par_grads():
        print("p1min", p1min.gradient if abs(p1min.gradient) > (10**-3) else 0)
        print("p1max", p1max.gradient if abs(p1max.gradient) > (10**-3) else 0)
        print("p2min", p2min.gradient if abs(p2min.gradient) > (10**-3) else 0)
        print("p2max", p2max.gradient if abs(p2max.gradient) > (10**-3) else 0)
        print("p3min", p3min.gradient if abs(p3min.gradient) > (10**-3) else 0)
        print("p3max", p3max.gradient if abs(p3max.gradient) > (10**-3) else 0)
        print("p4min", p4min.gradient if abs(p4min.gradient) > (10**-3) else 0)
        print("p4max", p4max.gradient if abs(p4max.gradient) > (10**-3) else 0)

    p1_max_est = prob1.solve(solver=cp.SCS, requires_grad=True)
    p2_max_est = prob2.solve(solver=cp.SCS, requires_grad=True)
    p3_max_est = prob3.solve(solver=cp.SCS, requires_grad=True)
    p4_max_est = prob4.solve(solver=cp.SCS, requires_grad=True)
    prob1.backward()
    print("Gradients of parameters with respect to p1=x")
    print_par_grads()
    prob2.backward()
    print("Gradients of parameters with respect to p2=x+y")
    print_par_grads()
    prob3.backward()
    print("Gradients of parameters with respect to p3=y")
    print_par_grads()
    prob4.backward()
    print("Gradients of parameters with respect to p4=0.7x + y")
    print_par_grads()
    return p1_max_est, p2_max_est, p3_max_est, p4_max_est


dep_max_est([0.5, 0.6, 0.2, 0.4], [0.1, 0.1, 0.2, 0.25])

print(v1.gradient)
print(v2.gradient)
