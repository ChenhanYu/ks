import numpy
from gsks import *

N = 10
m = 4
n = 3
d = 3

kerneltype = "Gaussian"
kernelscal = -1 / 2 * (0.5) * (0.5)

X     = numpy.random.rand( N, d ).astype( numpy.double )
w     = numpy.random.rand( N ).astype( numpy.double )
X2    = numpy.ndarray( shape = ( N ),    dtype = numpy.double )
u     = numpy.ndarray( shape = ( N ),    dtype = numpy.double )
alpha = numpy.ndarray( shape = ( m ),    dtype = numpy.int32 )
beta  = numpy.ndarray( shape = ( n ),    dtype = numpy.int32 )
mu    = numpy.ndarray( shape = ( m ),    dtype = numpy.int32 )
omega = numpy.ndarray( shape = ( n ),    dtype = numpy.int32 )

for i in range( m ): alpha[ i ] = i
for i in range( m ): mu[ i ] = i
for j in range( n ): beta[ j ] = j
for j in range( n ): omega[ j ] = j

for i in range( N ):
  X2[ i ] = 0.0
  for j in range( d ):
    X2[ i ] += X[ i, j ] * X[ i, j ]


for i in range( N ):
  u[ i ] = 0.0

print "gsks_ref:"
dgsks_ref( d, u, mu, X, X2, alpha, X, X2, beta, w, omega, type=kerneltype, scal=kernelscal )
print u

for i in range( N ):
  u[ i ] = 0.0

print "gsks:"
dgsks( d, u, mu, X, X2, alpha, X, X2, beta, w, omega, type=kerneltype, scal=kernelscal )
print u
