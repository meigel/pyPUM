from pypum.pum.function import Function

import numpy as np
import logging
logger = logging.getLogger(__name__)


class Monomial(Function):
    def __init__(self, degree=3):
        super(Monomial, self).__init__(dim=1, codim=1, vectorised=True)
        self._degree = degree
        self._f = np.poly1d([1] + [0] * (degree - 1))
        self._df = self._f.deriv(1) 
    
    def _f(self, x):
        return self._f(x)
    
    def _dx(self, x):
        return self._df(x)


class Spline(Function):
    """ Spline weight functions on [0,1].
        Note that the definitions are for [-1,1]. Thus, the input coordinates are transformed first.
    """
    
    def __init__(self, degree=3):
        assert 0 < degree and degree < 4
        super(Spline, self).__init__(dim=1, codim=1)
        self._degree = degree
        self._sf = {1:(self._s1, self._ds1), 2:(self._s2, self._ds2), 3:(self._s3, self._ds3)}
    
    def _f(self, x):
        return self._sf[self._degree][0](2 * x - 1)
    
    def _dx(self, x):
        return self._sf[self._degree][1](2 * x - 1)

    def _s1(self, x):
        x = np.abs(x)
        if x < 1:
            y = 1 - x
        else:
            y = 0
        return y

    def _ds1(self, x):
        if x < 0 and x > -1:
            y = 1
        elif x > 0 and x < 1:
            y = -1
        else:
            y = 0
        return y        

    def _s2(self, x):
        if x <= -1 or x >= 1:
            return 0

        x = np.abs((x + 1) / 2.0)
        if x <= 1 / 3.0:
            y = 6 * x * x
        elif x <= 2 / 3.0:
            y = 6 * (1 / 9.0 + 2 / 3.0 * (x - 1 / 3.0) - 2 * (x - 1 / 3.0) * (x - 1 / 3.0))
        elif x <= 1:
            y = 6 * ((1 - x) * (1 - x))
        else:
            assert False    # should never happen
        return y

    def _ds2(self, x):
        if x <= -1 or x >= 1:
            return 0

        x = np.abs((x + 1) / 2.0)
        if x <= 1 / 30:
            y = 0.5 * 12 * x
        elif x <= 2 / 3.0:
            y = 0.5 * 6 * (2 / 3.0 - 4 * (x - 1 / 3.0))
        elif x <= 1:
            y = 0.5 * -12 * (1 - x)
        else:
            assert False
        return y
    
    def _s3(self, x):
        if x <= -1 or x >= 1:
            return 0

        x = np.abs((x + 1) / 2.0)
        if x <= 1 / 4.0:
            y = 16 * x * x * x;
        elif x <= 2 / 4.0:
            y = 16 * (1 / 64.0 + 3 / 16.0 * (x - 1 / 4.0) + 3 / 4.0 * (x - 1 / 4.0) * (x - 1 / 4.0) - 3 * (x - 1 / 4.0) * (x - 1 / 4.0) * (x - 1 / 4.0))
        elif x <= 3 / 4.0:
            y = 16 * (1 / 64.0 + 3 / 16.0 * (3 / 4.0 - x) + 3 / 4.0 * (3 / 4.0 - x) * (3 / 4.0 - x) - 3.*(3 / 4.0 - x) * (3 / 4.0 - x) * (3 / 4.0 - x))
        elif x <= 1:
            y = 16 * ((1 - x) * (1 - x) * (1 - x))
        else:
            assert False
        return y

    def _ds3(self, x):
        if x <= -1 or x >= 1:
            return 0
        
        x = np.abs((x + 1) / 2.0)
        if x <= 1 / 4.0:
            y = 0.5 * 3 * 16 * x * x
        elif x <= 2 / 4.0:
            y = 0.5 * 16 * (3 / 16.0 + 6 / 4.0 * (x - 1 / 4.0) - 9 * (x - 1 / 4.0) * (x - 1 / 4.0))
        elif x <= 3 / 4.0:
            y = 0.5 * 16 * (-3 / 16.0 + -6 / 4.0 * (3 / 4.0 - x) + 9 * (3 / 4.0 - x) * (3 / 4.0 - x))
        elif x <= 1:
            s = -1 + 2 * int(x < 0)
            y = s * 0.5 * 3 * 16 * (1 - x) * (1 - x)
        else:
            assert False
        return y


# Gauss
#    void operator()(ublas::vector<DOUBLE> const& x, ublas::vector<DOUBLE>& y) const{
#    assert(x.size() == 1 && y.size() == 1);
#        y[0] = exp(-a*x[0]*x[0]);
#    }
#
#    void dx(ublas::vector<DOUBLE> const& x, ublas::vector<DOUBLE>& y) const{
#        y[0] = -2.*a*fabs(x[0])*exp(-a*x[0]*x[0]);
#    }

# cutoff (Brenner/Scott 4.1)
#     void operator()(ublas::vector<DOUBLE> const& x, ublas::vector<DOUBLE>& y) const{
#    assert(x.size() == 1 && y.size() == 1);
#        DOUBLE t = fabs(x[0]);
#        if(t<rho){
#            t /= rho;
#            t = 1. - t*t;
#            y[0] = exp(-(1./t));
#        }else
#            y[0] = 0.;
#    }
#
#    void dx(ublas::vector<DOUBLE> const& x, ublas::vector<DOUBLE>& y) const{
#    assert(x.size() == 1 && y.size() == 1);
#        DOUBLE t = fabs(x[0]);
#        if(t<rho){
#      DOUBLE tq = (t/rho);
#      tq *= tq;
#      DOUBLE dq = tq - 1.;
#      dq *= dq;
#            y[0] = ((x[0]<0.)?-1.:1.)*(-2. * t * exp(1./(tq - 1.))) / (rho*rho * dq);
#        }else
#            y[0] = 0.;
#    }

# tri-cube (Buhman p86)
#    void operator()(ublas::vector<DOUBLE> const& x, ublas::vector<DOUBLE>& y) const{
#    assert(x.size() == 1 && y.size() == 1);
#        DOUBLE t = fabs(x[0]);
#        t = 1. - t*t*t;
#        y[0] = (t>0.)?t*t*t:0.;
#    }
#
#    void dx(ublas::vector<DOUBLE> const& x, ublas::vector<DOUBLE>& y) const{
#        DOUBLE t = fabs(x[0]);
#        t = 1. - t*t*t;
#        y[0] = (t>0.)?-3.*t*t:0.;
#    }

