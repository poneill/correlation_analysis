
def fac(x):
    if x == 1:
       	return 1
    else:
	return x*fac(x-1)
def fib(n):
    if n < 2:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)
fib_monster = fib(10)

def bar(x):
    return 5