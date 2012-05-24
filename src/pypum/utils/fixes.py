"""
This module is for patching bugs in third party libraries. If there is
a bug in such a library the workaround should be implemented here, and
"monkey-patched" into the library, so that no conditional bug-fixing
code appears in one of the central spuq modules. This module is
automatically loaded from the main spuq.__init__ module so that it's
not necessary to explicitly load it in the module(s) that need(s) the
bug fix. 

For an example, see the code that "monkey-patches" a bug in the btdtri
function of scipy. It is "used" in spuq.stochastics.random_variable
(implicitly in the invcdf of the BetaRV). Without this module the unit
test for the BetaRV will fail for scipy version less than 0.9 (at
least for 0.7.2).
"""


import scipy.special

if True or scipy.version.version < "0.9.0":
    # still happens in 0.9.0 now (I thougt it was fixed there, need to check that later)
    old_btdtri = scipy.special.btdtri
    def my_btdtri(alpha, beta, q):
        if alpha == 0.5 and beta == 0.5 and q == 0.5:
            return 0.5
        else:
            return old_btdtri(alpha, beta, q)

    scipy.special.btdtri = my_btdtri
    print "INFO: scipy module patched"
