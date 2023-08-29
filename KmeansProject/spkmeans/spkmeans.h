# ifndef SPKMEANS_H_
# define SPKMEANS_H_

double* Kmeans(double *Kfirstpoints, double *points, int K, int iter, int m, int num, double eps);
double* Createwam(double * datapoints , int d , int num);
double* Createddg(double * datapoints , int d , int num );
double* Creategl(double * datapoints , int d , int num);
double** Createjac(double * matrix , int num );

# endif