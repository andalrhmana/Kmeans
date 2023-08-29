#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include "spkmeans.h"
#include <Python.h>

static PyObject* spk(PyObject *self, PyObject *args)
{
    PyObject *lst;
    PyObject *item;
    PyObject *firstKcntroids;
    PyObject *itemn;
    PyObject* python_val;
    PyObject* python_float;
    PyObject* val;
    int itr;
    double epsilon;
    int num;
    int K;
    int m;
    double *points;
    int i;
    int j;
    double cor;
    double *Kfirstpoints;
    int l;
    int k;
    double *avgvec;
    int h;
    int s;
    if (!PyArg_ParseTuple(args, "OOid", &lst, &firstKcntroids, &itr, &epsilon)) {
        return NULL;
    }
    num = PyObject_Length(lst);
    K = PyObject_Length(firstKcntroids);
    item = PyList_GetItem(lst, 0);
    m = PyObject_Length(item);
    points = calloc(num*m, sizeof(double)); 
    if (points == NULL){
        printf("An Error Has Occurred");
         exit(1);}
    Kfirstpoints = calloc(K*m, sizeof(double));
    if (Kfirstpoints == NULL){
        printf("An Error Has Occurred");
         exit(1);}     

    for (l = 0; l < num; l++) {
        item = PyList_GetItem(lst, l);
        for (k = 0; k < m; k++) {
            itemn = PyList_GetItem(item, k);
            cor = PyFloat_AsDouble(itemn);
            points[k + l*m] = cor;
        }
    }
    for (i = 0; i < K; i++) {
        item = PyList_GetItem(firstKcntroids, i);
        for (j = 0; j < m; j++) {
            itemn = PyList_GetItem(item, j);
            cor = PyFloat_AsDouble(itemn);
            Kfirstpoints[j + i*m] = cor;
        }
    }
    avgvec = Kmeans(Kfirstpoints, points, K, itr, m, num, epsilon);
    free(Kfirstpoints);
    free(points);

    python_val = PyList_New(K);
    for (s = 0; s < K; ++s)
    {
        val = PyList_New(m);
        for (h = 0; h < m; ++h){
            python_float = Py_BuildValue("d", avgvec[h + m*s]);
            PyList_SetItem(val, h, python_float);
        }    
        PyList_SetItem(python_val, s, val);
    }
    free(avgvec);
    return python_val;
}

static PyObject* wam(PyObject *self, PyObject *args)
{
    PyObject *lst;
    PyObject *item;
    PyObject *itemn;
    PyObject* python_val;
    PyObject* python_float;
    PyObject* val;
    int num;
    int m;
    double *points;
    double cor;
    int l;
    int k;
    double *avgvec;
    int h;
    int s;
    if (!PyArg_ParseTuple(args, "O", &lst)) {
        return NULL;
    }
    num = PyObject_Length(lst);
    item = PyList_GetItem(lst, 0);
    m = PyObject_Length(item);
    points = calloc(num*m, sizeof(double)); 
    if (points == NULL){
        printf("An Error Has Occurred");
         exit(1);}
    for (l = 0; l < num; l++) {
        item = PyList_GetItem(lst, l);
        for (k = 0; k < m; k++) {
            itemn = PyList_GetItem(item, k);
            cor = PyFloat_AsDouble(itemn);
            points[k + l*m] = cor;
        }
    }
    avgvec = Createwam(points, m, num);
    free(points);

    python_val = PyList_New(num);
    for (s = 0; s < num; ++s)
    {
        val = PyList_New(num);
        for (h = 0; h < num; ++h){
            python_float = Py_BuildValue("d", avgvec[h + num*s]);
            PyList_SetItem(val, h, python_float);
        }    
        PyList_SetItem(python_val, s, val);
    }
    free(avgvec);
    return python_val;
}

static PyObject* ddg(PyObject *self, PyObject *args)
{
    PyObject *lst;
    PyObject *item;
    PyObject *itemn;
    PyObject* python_val;
    PyObject* python_float;
    PyObject* val;
    int num;
    int m;
    double *points;
    double cor;
    int l;
    int k;
    double *avgvec;
    int h;
    int s;
    if (!PyArg_ParseTuple(args, "O", &lst)) {
        return NULL;
    }
    num = PyObject_Length(lst);
    item = PyList_GetItem(lst, 0);
    m = PyObject_Length(item);
    points = calloc(num*m, sizeof(double)); 
    if (points == NULL){
        printf("An Error Has Occurred");
         exit(1);}
    for (l = 0; l < num; l++) {
        item = PyList_GetItem(lst, l);
        for (k = 0; k < m; k++) {
            itemn = PyList_GetItem(item, k);
            cor = PyFloat_AsDouble(itemn);
            points[k + l*m] = cor;
        }
    }
    avgvec = Createddg(points, m, num);
    free(points);

    python_val = PyList_New(num);
    for (s = 0; s < num; ++s)
    {
        val = PyList_New(num);
        for (h = 0; h < num; ++h){
            python_float = Py_BuildValue("d", avgvec[h + num*s]);
            PyList_SetItem(val, h, python_float);
        }    
        PyList_SetItem(python_val, s, val);
    }
    free(avgvec);
    return python_val;
}

static PyObject* gl(PyObject *self, PyObject *args)
{
    PyObject *lst;
    PyObject *item;
    PyObject *itemn;
    PyObject* python_val;
    PyObject* python_float;
    PyObject* val;
    int num;
    int m;
    double *points;
    double cor;
    int l;
    int k;
    double *avgvec;
    int h;
    int s;
    if (!PyArg_ParseTuple(args, "O", &lst)) {
        return NULL;
    }
    num = PyObject_Length(lst);
    item = PyList_GetItem(lst, 0);
    m = PyObject_Length(item);
    points = calloc(num*m, sizeof(double)); 
    if (points == NULL){
        printf("An Error Has Occurred");
         exit(1);}
    for (l = 0; l < num; l++) {
        item = PyList_GetItem(lst, l);
        for (k = 0; k < m; k++) {
            itemn = PyList_GetItem(item, k);
            cor = PyFloat_AsDouble(itemn);
            points[k + l*m] = cor;
        }
    }
    avgvec = Creategl(points, m, num);
    free(points);

    python_val = PyList_New(num);
    for (s = 0; s < num; ++s)
    {
        val = PyList_New(num);
        for (h = 0; h < num; ++h){
            python_float = Py_BuildValue("d", avgvec[h + num*s]);
            PyList_SetItem(val, h, python_float);
        }    
        PyList_SetItem(python_val, s, val);
    }
    free(avgvec);
    return python_val;
}

static PyObject* jac(PyObject *self, PyObject *args)
{
    PyObject *lst;
    PyObject *item;
    PyObject *itemn;
    PyObject* python_val;
    PyObject* python_float;
    PyObject* val;
    PyObject* vectors;
    PyObject* vectorsEigenvalue;
    int num;
    double *matrix;
    double cor;
    int l;
    int k;
    double **jacobi;
    int h;
    int s;
    if (!PyArg_ParseTuple(args, "O", &lst)) {
        return NULL;
    }
    num = PyObject_Length(lst);
    matrix = calloc(num*num, sizeof(double)); 
    if (matrix == NULL){
        printf("An Error Has Occurred");
         exit(1);}
    for (l = 0; l < num; l++) {
        item = PyList_GetItem(lst, l);
        for (k = 0; k < num; k++) {
            itemn = PyList_GetItem(item, k);
            cor = PyFloat_AsDouble(itemn);
            matrix[k + l*num] = cor;
        }
    }
    jacobi = Createjac(matrix, num);
    free(matrix);
    python_val = PyList_New(2);
    vectors = PyList_New(num);
    vectorsEigenvalue = PyList_New(num);
    for (h = 0; h < num; ++h){
            python_float = Py_BuildValue("d", jacobi[0][h]);
            PyList_SetItem(vectorsEigenvalue, h, python_float);
    }    
    
    for (s = 0; s < num; ++s)
    {
        val = PyList_New(num);
        for (h = 0; h < num; ++h){
            python_float = Py_BuildValue("d", jacobi[1][h + num*s]);
            PyList_SetItem(val, h, python_float);
        }    
        PyList_SetItem(vectors, s, val);
    }
    PyList_SetItem(python_val, 0, vectorsEigenvalue);
    PyList_SetItem(python_val, 1, vectors);
    free(jacobi[0]);
    free(jacobi[1]);
    free(jacobi);
    return python_val;
}

static PyMethodDef spkmeansMethods[] = {
    {
      "spk",                   
      (PyCFunction) spk,
      METH_VARARGS,           
      PyDoc_STR("return ")
    },
    {
      "wam",                   
      (PyCFunction) wam,
      METH_VARARGS,           
      PyDoc_STR("return The Weighted Adjacency Matrix")
    },
    {
      "ddg",                   
      (PyCFunction) ddg,
      METH_VARARGS,           
      PyDoc_STR("return The Diagonal Degree Matrix")
    },
    {
      "gl",                   
      (PyCFunction) gl,
      METH_VARARGS,           
      PyDoc_STR("return The Graph Laplacian")
    },
    {
      "jac",                   
      (PyCFunction) jac,
      METH_VARARGS,           
      PyDoc_STR(" Finding Eigenvalues and Eigenvectors for a matrix")
    }, 
    {NULL, NULL, 0, NULL}     
};

static struct PyModuleDef spkmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", 
    NULL, 
    -1,  
    spkmeansMethods 
};

PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&spkmeansmodule);
    if (!m) {
        return NULL;
    }
    return m;
}