import os
import ctypes
from numpy import *
import numpy



# ---------------------------------------------------------------------------
# @breif
#        The first 12 arguments are required for all kernels. The rest
#        of the arguments are stored in dictionary called **options.
#        options.get("type") indicates the kernel type.
#        options.get("scal") indicates the multiplication scalar.
#        options.get("cons") indicates the shifting constant. 
#        options.get("powe") indicates the power degree in laplace and poly. 
#        options.get("h")    indicates the variable bandwidth vector.       
# ---------------------------------------------------------------------------
def dgsks_ref( d, u, mu, XA, XA2, alpha, XB, XB2, beta, w, omega, **options ):
  M = XA.shape[ 0 ]
  N = XB.shape[ 0 ]
  m = alpha.size
  n = beta.size
  # Range checking
  assert XA.shape[ 1 ] == XB.shape[ 1 ]
  assert M >= m
  assert N >= n
  assert XA2.size == M
  assert XB2.size == N
  # Type checking
  assert XA.dtype == numpy.double
  assert XB.dtype == numpy.double
  assert XA2.dtype == numpy.double
  assert XB2.dtype == numpy.double
  assert alpha.dtype == numpy.int32
  assert beta.dtype == numpy.int32
  assert mu.dtype == numpy.int32
  assert omega.dtype == numpy.int32

  kertype = 0
  kerscal = 1.0
  kercons = 0.0
  kerpowe = 1.0
  kerh    = numpy.ndarray( shape = ( 1 ), dtype = numpy.double )

  # Differentiate between different kernels.
  if options.get("type") == "Gaussian":
    print "Gussian"
    kertype = 0
    kerscal = options.get("scal")
  elif options.get("type") == "Polynomial":
    print "Polynomial"
    kertype = 1
    kerscal = options.get("scal")
    kerscal = options.get("cons")
    kerpowe = options.get("powe")
  elif options.get("type") == "Laplace":
    print "Laplace"
    kertype = 2
  elif options.get("type") == "Variable_Bandwidth":
    print "Variable Bandwidth"
    kertype = 3
    kerh    = options.get("h")
  elif options.get("type") == "Tanh":
    print "Tanh"
    kertype = 4
    kerscal = options.get("scal")
    kerscal = options.get("cons")
  elif options.get("type") == "Quartic":
    print "Quartic"
    kertype = 5
  elif options.get("type") == "Multiquadratic":
    print "Multiquadratic"
    kertype = 6
    kerscal = options.get("cons")
  elif options.get("type") == "Epanechnikov":
    print "Epanechnikov"
    kertype = 7
  else:
    print "This kernel option is not suppurted."
    return

  # Load the library.
  libgsks_path = os.environ.get( 'GSKS_DIR' ) + '/lib/libgsks.so'
  libgsks = ctypes.cdll.LoadLibrary( libgsks_path )
  libgsks.dgsks_ref_wrapper(
    ctypes.c_int( m ),
    ctypes.c_int( n ),
    ctypes.c_int( d ),
    ctypes.c_void_p( u.ctypes.data ),
    ctypes.c_void_p( mu.ctypes.data ),
    ctypes.c_void_p( XA.ctypes.data ),
    ctypes.c_void_p( XA2.ctypes.data ),
    ctypes.c_void_p( alpha.ctypes.data ),
    ctypes.c_void_p( XB.ctypes.data ),
    ctypes.c_void_p( XB2.ctypes.data ),
    ctypes.c_void_p( beta.ctypes.data ),
    ctypes.c_void_p( w.ctypes.data ),
    ctypes.c_void_p( omega.ctypes.data ),
    ctypes.c_int( kertype ),
    ctypes.c_double( kerscal ),
    ctypes.c_double( kercons ),
    ctypes.c_double( kerpowe ),
    ctypes.c_void_p( kerh.ctypes.data )
    )
  return



#
#
#
def dgsks( d, u, mu, XA, XA2, alpha, XB, XB2, beta, w, omega, **options ):
  M = XA.shape[ 0 ]
  N = XB.shape[ 0 ]
  m = alpha.size
  n = beta.size
  # Range checking
  assert XA.shape[ 1 ] == XB.shape[ 1 ]
  assert M >= m
  assert N >= n
  assert XA2.size == M
  assert XB2.size == N
  # Type checking
  assert XA.dtype == numpy.double
  assert XB.dtype == numpy.double
  assert XA2.dtype == numpy.double
  assert XB2.dtype == numpy.double
  assert alpha.dtype == numpy.int32
  assert beta.dtype == numpy.int32
  assert mu.dtype == numpy.int32
  assert omega.dtype == numpy.int32

  kertype = 0
  kerscal = 1.0
  kercons = 0.0
  kerpowe = 1.0
  kerh    = numpy.ndarray( shape = ( 1 ), dtype = numpy.double )

  # Differentiate between different kernels.
  if options.get("type") == "Gaussian":
    print "Gussian"
    kertype = 0
    kerscal = options.get("scal")
  elif options.get("type") == "Polynomial":
    print "Polynomial"
    kertype = 1
    kerscal = options.get("scal")
    kerscal = options.get("cons")
    kerpowe = options.get("powe")
  elif options.get("type") == "Laplace":
    print "Laplace"
    kertype = 2
  elif options.get("type") == "Variable_Bandwidth":
    print "Variable Bandwidth"
    kertype = 3
    kerh    = options.get("h")
  elif options.get("type") == "Tanh":
    print "Tanh"
    kertype = 4
    kerscal = options.get("scal")
    kerscal = options.get("cons")
  elif options.get("type") == "Quartic":
    print "Quartic"
    kertype = 5
  elif options.get("type") == "Multiquadratic":
    print "Multiquadratic"
    kertype = 6
    kerscal = options.get("cons")
  elif options.get("type") == "Epanechnikov":
    print "Epanechnikov"
    kertype = 7
  else:
    print "This kernel option is not suppurted."
    return

  # Load the library.
  libgsks_path = os.environ.get( 'GSKS_DIR' ) + '/lib/libgsks.so'
  libgsks = ctypes.cdll.LoadLibrary( libgsks_path )
  libgsks.dgsks_wrapper(
    ctypes.c_int( m ),
    ctypes.c_int( n ),
    ctypes.c_int( d ),
    ctypes.c_void_p( u.ctypes.data ),
    ctypes.c_void_p( mu.ctypes.data ),
    ctypes.c_void_p( XA.ctypes.data ),
    ctypes.c_void_p( XA2.ctypes.data ),
    ctypes.c_void_p( alpha.ctypes.data ),
    ctypes.c_void_p( XB.ctypes.data ),
    ctypes.c_void_p( XB2.ctypes.data ),
    ctypes.c_void_p( beta.ctypes.data ),
    ctypes.c_void_p( w.ctypes.data ),
    ctypes.c_void_p( omega.ctypes.data ),
    ctypes.c_int( kertype ),
    ctypes.c_double( kerscal ),
    ctypes.c_double( kercons ),
    ctypes.c_double( kerpowe ),
    ctypes.c_void_p( kerh.ctypes.data )
    )
  return
