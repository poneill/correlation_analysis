"""
Compute power expansions for two state replicator mutator equation.
"""

from sympy import *
x,y,t = var('x'),var('y'),var('t')
alpha,beta,gamma,delta = var('alpha'),var('beta'),var('gamma'),var('delta')
psi0 = x

def A(expr):
    return alpha * (x**2-x)*diff(expr,x)

def B(expr):
    return beta * (x*y-y)*diff(expr,y)

def C(expr):
    return gamma * (x*y-x)*diff(expr,x)

def D(expr):
    return delta * (y**2-y)*diff(expr,y)
    
def H(expr):
    return A(expr) + B(expr) + C(expr) + D(expr)

def psi(n):
    expr = psi0
    ans = expr
    for i in range(1,n+1):
        expr = H(expr)*t/i
        ans += expr
    return ans

def prob(psin,i,j):
    return (diff(diff(psin,x,i),y,j)/(factorial(i)*factorial(j))).subs(x,0).subs(y,0)

def parametrize(expr,a,b,c,d):
    return expr.subs(alpha,a).subs(beta,b).subs(gamma,c).subs(delta,d)
