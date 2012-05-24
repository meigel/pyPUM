"""ForgetfulVector can be indexed like vectors but "forget" older
values, which are not needed any more to conserve memory. The user has
to specify how many entries to keep. The ForgetfulVector automatically
grows in direction to higher indices. This makes it very easy to
implement algorithms that are defined iteratively, without bothering
with temp variables or whether one can overwrite existing variables.

The Fibonacci numbers need to go back 2 values into the past in their
definition; so use a ForgetfulVector with argument 2, then set the 2
initial values:
>>> f=ForgetfulVector(2)
>>> f[0]=1
>>> f[-1]=0
>>> print 0, f[0]
0 1

Now compute the next few numbers:
>>> for i in range(1,8): 
...   f[i]=f[i-1]+f[i-2]
...   print i, f[i]
1 1
2 2
3 3
4 5
5 8
6 13
7 21

Now we can check that only the values for the last two indices 6 and 7
are stored:
>>> print f[5]
Traceback (most recent call last):
    ...
IndexError: index out of range: 5 not in [6, 7]
>>> print f[6]
13
>>> print f[7]
21
>>> print f[8]
Traceback (most recent call last):
    ...
IndexError: index out of range: 8 not in [6, 7]
"""


class ForgetfulVector(object):
    def __init__(self, remember, start=0):
        self.rem=remember
        self.curr=start
        self.items=self.rem*[None]

    def __getitem__(self,k):
        if not self.curr-self.rem<k<=self.curr:
            raise IndexError("index out of range: %d not in [%d, %d]" % 
                             (k, self.curr-self.rem+1, self.curr))
        return self.items[self.curr-k]

    def __setitem__(self,k,item):
        if not self.curr-self.rem<k<=self.curr+1:
            raise IndexError("index out of range: %d not in [%d, %d]" % 
                             (k, self.curr-self.rem+1, self.curr+1))
        elif k==self.curr+1:
            self.items = [item]+self.items[:-1]
            self.curr = self.curr+1
        else:
            self.items[self.curr-k] = item

    def __repr__(self):
        s=[]
        for d in xrange(self.rem):
            s=s+[str(self.curr-d)+": "+str(self.items[d])];
        return ", ".join(s)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
