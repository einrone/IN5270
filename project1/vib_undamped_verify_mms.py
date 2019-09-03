import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
V, t, I, w, dt, a, b= sym.symbols('V t I w dt a b')  # global symbols
f = None  # global variable for the source term in the ODE

def ode_source_term(u):
    """Return the terms in the ODE that the source term
    must balance, here u'' + w**2*u.
    u is symbolic Python function of t."""
    return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
    """Return the residual of the discrete eq. with u inserted."""
    R =  DtDt(u,dt) + w**2*u(t) - f
    return sym.simplify(R)

def residual_discrete_eq_step1(u):
    """Return the residual of the discrete eq. at the first
    step with u inserted."""

    u1 = (dt**2*f.subs(t,0))/2 + I - (I*w**2*dt**2)/2 + V*dt

    R = u(t).subs(t,dt) - u1
    return sym.simplify(R)

def DtDt(u, dt):
    """Return 2nd-order finite difference for u_tt.
    u is a symbolic Python function of t.
    """
    return (u(t).subs(t, t+dt) - 2*u(t) + u(t).subs(t, t-dt))/dt**2

def main(u):
    """
    Given some chosen solution u (as a function of t, implemented
    as a Python function), use the method of manufactured solutions
    to compute the source term f, and check if u also solves
    the discrete equations.
    """
    print ('=== Testing exact solution: %s ===' % u(t))
    print (("Initial conditions u(0)=%s, u'(0)=%s:") %
          (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0)))

    # Method of manufactured solution requires fitting f
    global f  # source term in the ODE
    f = sym.simplify(ode_source_term(u))

    # Residual in discrete equations (should be 0)
    print (('residual step1:'), residual_discrete_eq_step1(u))
    print (('residual:'), residual_discrete_eq(u))

def linear():
    main(lambda t: V*t + I)

def quadratic():
    main(lambda t: a*t**2 + V*t + I)

def cubic():
    main(lambda t: a*t**3 + b*t**2 + V*t + I)

def solver(u, u_e, n, T, I_set, V_set, a_set, b_set, w_set):
    dt = T/n
    time = np.linspace(0, T, n+1)

    global I, V, a, b, w
    I = I_set
    V = V_set
    a = a_set
    b = b_set
    w = w_set

    f = ode_source_term(u_e)
    f = sym.lambdify(t,f)


    u[0] = I
    u[1] = 0.5*f(0)*dt**2 + I*(1- 0.5*w**2*dt**2) - V*dt


    for i in range(1,n):
        u[i+1] = 2*u[i] - u[i-1] + f(time[i])*dt**2 -w**2*u[i]*dt**2

    return time, u





if __name__ == '__main__':
    linear()
    quadratic()
    cubic()

    N = 100000
    u = np.zeros(N+1)
    time, u = solver(u,lambda t: a*t**2 + V*t + I , N, 1, 0.5, 0.3, 1, 0.2, 0.1)

    exact = lambda t: a*t**2 + V*t + I
    plt.plot(time, u)
    plt.plot(time, exact(time))
    plt.legend(['numerical','exact'])
    plt.title("solution of a quadratic polynomial")
    plt.xlabel("time [$t$]")
    plt.ylabel("solution $u(t)$")
    plt.show()
