def coefficients(lamb,mu,psi0,c,num_terms):
    psinm1 = 0
    psin = psi0
    coeffs = [psin]
    def diff_eq(psin,psinm1,n):
        return ((c + lamb + mu*n)*psin - lamb*psinm1)/(mu*(n+1))
    for _ in range(num_terms):
        
