from sklearn.tree._utils cimport log

cimport numpy as cnp
from sklearn.tree._criterion cimport SIZE_t, ClassificationCriterion, RegressionCriterion

cdef class SimultaneousCriterion(RegressionCriterion):
    cdef __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples, cnp.ndarray[SIZE_t, ndim=1] n_classes):
        self.clf = ClassificationCriterion(n_outputs, n_classes)
        self.reg = RegressionCriterion(n_outputs, n_samples)

    cdef double node_impurity(self) nogil:
        ...

    cdef void children_impurity(self, double * impurity_left,
                                double * impurity_right) nogil:
        ...