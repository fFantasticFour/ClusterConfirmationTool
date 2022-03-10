
from cpython.pycapsule cimport (PyCapsule_New,
                                PyCapsule_GetPointer)
from cpython.mem cimport PyMem_Malloc,  PyMem_Free
from libc.math cimport exp, sqrt, M_PI, log,cos, atan, atanh
from libc.math cimport tgamma
import scipy

cdef double c_dblintegrand(int n, double* xx, void* user_data):
    """The integrand, written in Cython"""
    # Extract a.
    # Cython uses array access syntax for pointer dereferencing!
    cdef double alpha = (<double*>user_data)[0]
    cdef double beta = (<double*>user_data)[1]
    cdef double sigma2 = (<double*>user_data)[2]
    cdef double R = (<double*>user_data)[3]
    
    r_prime=xx[1]
    theta_prime=xx[0]
    
    F = beta**alpha/tgamma(alpha)*r_prime**(alpha-1)*exp(-beta*r_prime)

#    F = exp(-(r_prime)**2/2/sigma1**2)
    
    r_dbl_prime=sqrt(r_prime**2+R**2-2*R*r_prime*cos(theta_prime))
    
    G = exp(-(r_dbl_prime)**2/2/sigma2**2)/sigma2**2
    
    return F*G
#
# Now comes some classic C-style housekeeping
#

cdef object pack_a(double alpha, double beta, double sigma2, double R):
    """Wrap 'a' in a PyCapsule for transport."""
    # Allocate memory where 'a' will be saved for the time being
    cdef double* a_ptr = <double*> PyMem_Malloc(4*sizeof(double))
    a_ptr[0] = alpha
    a_ptr[1] = beta
    a_ptr[2] = sigma2
    a_ptr[3] = R
    return PyCapsule_New(<void*>a_ptr, NULL, free_a)

cdef void free_a(capsule):
    """Free the memory our value is using up."""
    PyMem_Free(PyCapsule_GetPointer(capsule, NULL))

def get_low_level_callable(double alpha, double beta, double sigma2, double R):
    # scipy.LowLevelCallable expects the function signature to
    # appear as the "name" of the capsule
    func_capsule = PyCapsule_New(<void*>c_dblintegrand,
                                 "double (int, double *, void *)",
                                 NULL)
    data_capsule = pack_a(alpha,beta,sigma2,R)
    
    return scipy.LowLevelCallable(func_capsule, data_capsule)
