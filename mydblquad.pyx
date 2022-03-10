from cpython.pycapsule cimport (PyCapsule_New,
                                PyCapsule_GetPointer)
from cpython.mem cimport PyMem_Malloc,  PyMem_Free
from libc.math cimport exp, sqrt, M_PI, log,cos, atan, atanh
import scipy

cdef double c_dblintegrand(int n, double* xx, void* user_data):
    """The integrand, written in Cython"""
    # Extract a.
    # Cython uses array access syntax for pointer dereferencing!
    cdef double sigma = (<double*>user_data)[0]
    cdef double r_s = (<double*>user_data)[1]
    cdef double r = (<double*>user_data)[2]
    
    R=xx[1]
    theta=xx[0]
    
    G=exp(-R**2/2/sigma**2)
    
    f=0
    
    r_prime=sqrt(R**2+r**2-2*R*r*cos(theta))
    x=r_prime/r_s
    
    if x>1:
        arg=sqrt( (x-1)/(x+1) )
        f=1-2/sqrt(x**2-1)*atan(arg)
    if x<1:
        arg=sqrt( (1-x)/(1+x) )
        f=1-2/sqrt(1-x**2)*atanh(arg)
    
    Sigma = 1/(x**2-1)*f

    if x<1e-3/0.15:
        x2=1e-3/0.15
        arg=sqrt( (1-x2)/(1+x2) )
        f=1-2/sqrt(1-x2**2)*atanh(arg)
        Sigma = 1/(x2**2-1)*f
    
    return R*Sigma*G/2/M_PI/sigma**2
#
# Now comes some classic C-style housekeeping
#

cdef object pack_a(double sigma, double r_s, double r):
    """Wrap 'a' in a PyCapsule for transport."""
    # Allocate memory where 'a' will be saved for the time being
    cdef double* a_ptr = <double*> PyMem_Malloc(3*sizeof(double))
    a_ptr[0] = sigma
    a_ptr[1] = r_s
    a_ptr[2] = r
    return PyCapsule_New(<void*>a_ptr, NULL, free_a)

cdef void free_a(capsule):
    """Free the memory our value is using up."""
    PyMem_Free(PyCapsule_GetPointer(capsule, NULL))

def get_low_level_callable(double sigma, double r_s, double r):
    # scipy.LowLevelCallable expects the function signature to
    # appear as the "name" of the capsule
    func_capsule = PyCapsule_New(<void*>c_dblintegrand,
                                 "double (int, double *, void *)",
                                 NULL)
    data_capsule = pack_a(sigma,r_s,r)
    
    return scipy.LowLevelCallable(func_capsule, data_capsule)
