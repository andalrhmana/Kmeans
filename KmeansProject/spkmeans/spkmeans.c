#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>

double Euc(double *p ,double *q , int d){
    double sum = 0; 
    int i;
    for ( i=0 ; i< d ; i++){
        sum+= (p[i] - q[i]) * (p[i] - q[i]); }

    return sum;}

double Euc1(double *p ,double *q , int d){
    double sum = 0; 
    int i;
    for ( i=0 ; i< d ; i++){
        sum+= (p[i] - q[i]) * (p[i] - q[i]); }
    return sqrt(sum);}


int sumvectors( double *p ,double *q ,int d){
    int i;
    for (i=0 ; i<d ; i++){
        *(p + i) = (*(p + i) + *(q + i));}
    return 0;}

 
double *Kmeans(double *Kfirstpoints, double *points, int K, int iter, int m, int num, double eps){
    double epsilon = eps;
    int d = m;
    int numofpoints = num;
    double *p;
    double *G;
    double *H;
    int *numInCluster; 
    int it;
    int boo;
    int yy;
    int w;
    int r;
    int o;
    int i ;
    int e;
    int mm;
    G = calloc(K*d, sizeof(double));  /*error###########*/
    if (G == NULL){
        printf("An Error Has Occurred");
        exit(1);}

    H = calloc(K*d, sizeof(double));  /*error###########*/
    if (H == NULL){
        printf("An Error Has Occurred");
         exit(1);}

    for (yy=0 ; yy<K ; yy++){
        int ii;
        for(ii = 0; ii < d;ii++){
            H[ii + yy*d] = Kfirstpoints[ii + yy*d]; }}

    it = 0;
    boo= 0; 
    
    while((it < iter ) &  (!boo) ){
        p = (double *)calloc(K*d, sizeof(double));   /*error###########*/
        if (p == NULL){
            printf("An Error Has Occurred");
            exit (1);}
        for( w = 0; w < K*d; w++){
            p[w] = 0.0; }
        numInCluster = (int *) calloc (K , sizeof(int)); 
        if (numInCluster == NULL){
            printf("An Error Has Occurred");
            exit(1) ;}
        for( mm = 0; mm < K; mm++){
            numInCluster[mm] = 0; }

        for( r=0; r <numofpoints ; r++){
            double mindis;
            int idmindis = 0; 
            int c; 
            mindis = Euc1(points + r*d, H , d);  
            for (c=0 ; c<K ;  c++){
                double Eucdis = Euc1(points + r*d,H + d*c, d);
                if (Eucdis<= mindis){
                    mindis = Eucdis;
                    idmindis = c;   }}
            
            sumvectors(p + idmindis*d , points + r*d, d); 
            numInCluster[idmindis]++;    }
         
        for (o=0 ; o<K ; o++){
            int u; 
            for ( u=0 ; u<d ; u++){
                G[u + o*d] = H[u + o*d]; 
                H[u + o*d] = p[u + o*d]; }}
        
        for (i=0; i< K ; i++){
            int j;
            for ( j=0 ; j<d ; j++){

                H[j + i*d] = H[j + i*d] / numInCluster[i] ; }}

        boo = 1;
          
        for ( e=0 ; e<K ; e++){
            double DELTA= Euc1( G + e*d , H + e*d , d ); 
            if (DELTA>= epsilon){
                boo = 0; }    }
        
        it++;
        free(p);
        free(numInCluster);
    }
    free(G);
    return H;    
}


double* Createwam(double * datapoints , int d , int num) { 
    double * W;
    double res; 
    int i=0; 
    int j; 

    W = calloc(num * num , sizeof(double)); 
    if (W == NULL){
        printf("An Error Has Occurred");
        exit(1);}

    for(i=0 ; i<num ; i++){
        for (j=i+1 ; j<num ; j++){
            res = Euc(datapoints + i*d , datapoints + j*d, d);
            res= -0.5 * res;
            res = exp(res);
            W[i*num + j] = res;
            W[j*num + i] = res; 
        }
    }
    for(i=0 ; i<num ; i++){
        W[i*num + i] = 0;
    }
    return W;
}

double* Createddg(double * datapoints , int d , int num ){
    int i=0;
    int j=0;
    double sum=0;
    double * W= Createwam(datapoints, d, num);
    double * D = calloc(num * num , sizeof(double)); 
    if (D == NULL){
        printf("An Error Has Occurred");
        exit(1);}
    for (i=0 ; i<num ; i++){
        for (j=0 ; j<num ; j++){
            D[i*num + j] = 0; 
            sum += W[i*num + j];
        }
        D[i*num + i]= sum;
        sum=0; 
    }
    free(W);
    return D;
}

double* Creategl(double * datapoints , int d , int num){
    int i=0;
    int j=0;
    double sum =0; 
    double * L;
    double * W;
    W= Createwam(datapoints, d, num);
    L = calloc(num * num , sizeof(double)); 
    if (L == NULL){
        printf("An Error Has Occurred");
        exit(1);}

    for (i=0 ; i<num ; i++){
        for (j=0 ; j < num ; j++){
            L[i*num +j] = -W[i*num + j]; 
            sum += W[i*num+j];
        }
        L[i*num + i] += sum;
        sum = 0; 
    }
    free(W); 
    return L;}


    double* kanoni(double * matrix , int num ){
        int z=0; 
        int x=0;
        for (z=0 ; z<num ; z++){
            for (x=0 ; x<num ; x++){
                matrix[z*num +x] = 0;
            }
            matrix[z*num+z] = 1;
        }
        return matrix; 
    }

    void multi(double * mat1 , double *mat2 , int num){
        int col;
        int row;
        int ind; 
        double sum =0;
        double * new = calloc(num * num , sizeof(double)); 
        if (new == NULL){
            printf("An Error Has Occurred");
            exit(1);}
        
        for (row =0; row <num ; row++){
            for (col=0 ; col <num ; col++){
                for (ind =0 ; ind<num ; ind++){
                    sum += mat1[row +ind] *mat2[col +num *ind];  }
                new[row * num + col] = sum;
                sum=0; 
            }
        }
         for (row =0; row <num ; row++){
            for (col=0 ; col <num ; col++){
                mat1[row * num + col] = new[row * num + col];
            }
        }
        free(new);
    }

    int convergence( double * A, int num ){
        int i=0, j=0, sum=0 ; 
        for(i =0 ; i<num ; i++){
            for (j=i+1 ; j<num ; j++){
                /* (2*) BECAUSE ITS SYMETRIC*/

                sum += 2 * (pow(A[i*num +j] ,2));
            }
        }
        return sum;
    }


    
    
/*
 * Input: A,B: 2D-array of doubles. n: dimension.
 * Output: A*B. (* is matrices multiplication).
 */
double* matMult(double *A, double *B, int n){
    double *C;
    double res;
    int i, j, k;
    C=calloc(n*n, sizeof(double *));
    assert(C!=NULL);
    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
            res=0;
            for(k=0; k<n; k++){
                res+=(double)(A[i*n + k]*B[k*n + j]);
                C[i*n + j]=res;
                if(res==-0){
                    C[i*n + j]=0;
                }
            }
        }
    }
    return C;
}





/*
 * Input: A: 2D-array of doubles. n, d: dimensions.
 * Output: At: the transpose of A.
 */
double* createTranspose(double *A, int n) {
    double *transMat;
    int i, j;
    transMat=calloc(n*n, sizeof(double *));
    assert(transMat!=NULL);
    for(i=0;i<n;i++) {
        for(j=0;j<n;j++) {
            transMat[j*n + i] = A[i*n + j];
        }
    }
    return transMat;
}






/*This section contains the algorithm's main functions*/


/*
 * Input: M Matrix to be copied, n - dimension. 
 * Output: a copy of the provided matrix. 
 */
double* createCopy(double *M, int n) {
    double *res;
    int i, j;
    res = calloc(n*n, sizeof(double *));
    assert(res!=NULL);
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            res[i*n +j] = M[i*n +j];
        }
    }
    return res;
}

/*
 * Input: n: dimension.
 * Output: Identity matrix.
 */
double* createI(int n){
    double *I;
    int i,j;
    I=calloc(n*n, sizeof(double *));
    assert(I!=NULL);
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            I[i*n +j] = (double)0;
        }
    } 
    for(i=0; i<n; i++){
        I[i*n +i]=(double)1;
    }
    return I;
}




/*
 * Input: Lnorm - Lnorm Matrix, n - dimention.
 * Output: P - Rotation Matrix.
 */
double* createRotMat(double *Lnorm, int n) {
    double *P;
    int i, j;
    int imax = 0;
    int jmax = 1;
    double max, c, s, theta, t=0;
    max = 0;
    P = createI(n);
    assert(P!=NULL);
    for(i = 0; i < n; i++) {
        for(j = i + 1; j < n; j++) {
            if (fabs(Lnorm[i*n +j]) > fabs(max)) {
                max = Lnorm[i*n +j];
                imax = i;
                jmax = j;
            }
        }
    }
    theta = ((Lnorm[jmax*n + jmax] - Lnorm[imax*n + imax]) / (2*max));
    if(theta >= 0) {
        t = ( 1 / (fabs(theta) + sqrt(1+(theta*theta))) );
    } 
    else if(theta < 0) {
        t = ( (-1) / (fabs(theta) + sqrt(1+(theta*theta))) );
    }
    c = ( 1 / (sqrt((t*t) + 1)) );
    s = t*c;
    P[imax*n + jmax] = s;
    P[jmax*n + imax] = (-1)*s;
    P[imax*n + imax] = c;
    P[jmax*n + jmax] = c;
    return P;
}


/*
 * Input: Matrix A, Matrix P - Rotation Matrix
 * Output: A' = TransposeP * A * P 
 */
double* createAtag(double *A, double *P, int n) {
    double *Atag, *transP, *B;
    transP = createTranspose(P, n);
    assert(transP!=NULL);
    B = matMult(transP, A, n);
    assert(B!=NULL);
    Atag = matMult(B, P, n);
    assert(Atag!=NULL);
    free(B);
    free(transP);
    return Atag;
}


/*
 * Input: mat - Matrix, n - dimension
 * Output: square of the off-diagonal elements of the provided matrix.
 */ 
double OFF(double *mat, int n) {
    int i, j;
    double res = 0.0;
    for(i = 0;i < n;i++) {
        for(j = 0;j < n; j++) {
            if(i!=j){
                res = res + pow(mat[i*n + j],2);
            }
        }
    }
    return res;
}

/*
 * Input: n - dimension, M1 - first matrix, M2 - second matrix.
 * Output: This function calculates the convergence between two matrices M1 and M2. Calls the OFF function on each one of the matrices. 
 */
double calcConvergence(int n, double *M1, double *M2) {
    double offM1 = 0, offM2 = 0;
    double res;
    offM1 = OFF(M1, n);
    offM2 = OFF(M2, n);
    res = (offM1 - offM2);
    return res;
}

/*
 * Input: Matrix A, 
 * Output: return Jacobi Matrix as "vectors" matrix and "vals" is eigenvalues vector
 */
double** Createjac(double *A, int n) {
    int i, iterN=0,j;
    double eps, convergence;
    double *Atag, *P, *vectors;
    double *helper,*A2;
    double **res, *eigenvalue;
    res = calloc(2 , sizeof(double*)); 
    assert(res!=NULL);
    eps = pow(10, -5);
    A2 = createCopy(A, n);
    vectors = createI(n);
    assert(A2!=NULL);
    assert(vectors!=NULL);
    while (iterN<100) {
        P = createRotMat(A2, n);
        assert(P!=NULL);
        Atag = createAtag(A2, P, n);
        assert(Atag!=NULL);
        helper = createCopy(vectors, n); 
        assert(helper!=NULL); 
        free(vectors);
        vectors = matMult(helper, P, n);
        assert(vectors!=NULL);
        convergence = calcConvergence(n, A2, Atag);
        if(convergence <= eps) {
            free(A2);
            A2 = createCopy(Atag, n);
            assert(A2!=NULL);
            free(Atag);
            free(P);
            free(helper);
            break;
        }
        free(A2);
        A2 = createCopy(Atag, n);
        assert(A2!=NULL);
        iterN = iterN + 1;
        free(Atag);
        free(P);
        free(helper);
    }
    eigenvalue = calloc(n , sizeof(double)); 
    assert(eigenvalue!=NULL);
    for(i = 0; i<n; i++) {
        if (A2[i*n + i] == (-0.0)) {
            eigenvalue[i] = (double)0;
            for(j = 0; j<n; j++){
                vectors[j*n + i] = (-1)*vectors[j*n + i];
            }
        }else{
            eigenvalue[i] = A2[i*n + i];
        }    
    }
    res[0] = eigenvalue;
    res[1] = vectors;
    free(A2);
    return res;
}



struct linked_list
{
double h;
struct linked_list *next;
};
typedef struct linked_list ELEMENT;
typedef ELEMENT* LINK;


void delete_list( LINK head ){
    if ( head != NULL ){   
        delete_list( head->next);
        free( head );   }    }
 

int main(int argc,char **arg){
    double x;
    char c;
    int first = 0;
    int d = 0;
    int m = 0;
    int num = 0;
    LINK head = NULL, tail = NULL;
    double *p;
    int i = 0;
    int gg;
    double **avgvec;
    double *result;
    int n;
    int k;
    char *word;
    char *filename;
    FILE *fp;
    
    if (argc < 3) {
        exit(1);
    }
    word = arg[1];


    if(stdin == NULL){   /*error###########*/
        printf("An Error Has Occurred");
        exit(0);}
    filename = arg[2];
    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    while(fscanf(fp,"%lf%c",&x,&c) == 2){
        if(first == 0){
            head = (ELEMENT* )malloc( sizeof( ELEMENT ) );
            if (head == NULL){
                printf("An Error Has Occurred");
                exit(1);}
            head -> h = x;
            tail = head;
            first = 1;
            d++;
            num++;}
        
        else{
            tail->next = (ELEMENT*)malloc(sizeof( ELEMENT ) );
            tail = tail->next;
            if (tail == NULL){
                printf("An Error Has Occurred");
                exit(1);}
            tail->h = x;
            tail->next = NULL;
            d++;
            num++;}

        if(c == '\n'){
            m = d;
            d = 0;
        }
        
    }
    
    tail->next = (ELEMENT* )malloc( sizeof( ELEMENT ) );
    tail = tail->next;
     if (tail == NULL){
        printf("An Error Has Occurred");
        exit(1);}
    tail->h = x;
    tail->next = NULL;
    d++; 
    num++;
    /*if ( K >= num / m){
        printf("Invalid number of clusters!");
        exit(1);}*/
    

    tail = head;
    p = calloc(num + 1, sizeof(double));   /*error########### */
    if (p == NULL){
        printf("An Error Has Occurred");
        exit(1);}
    
    while(tail != NULL){
        p[i] = tail->h;
        tail = tail->next;
        i++;}

    delete_list(head);
    if(strcmp(word, "jacobi") == 0){
    avgvec = Createjac(p, m); /*################*/
    n = (int)num/m ;
    free(p);
    printf("%.4f",avgvec[0][0]);
    for(k = 1;k < n;k++){
            printf(",%.4f",avgvec[0][k]); }
    printf("\n");
        
    for( gg = 0; gg < n; gg++){
        printf("%.4f",avgvec[1][gg*n]);
        for(k = 1;k < n;k++){
            printf(",%.4f",avgvec[1][gg*n + k]); }
        printf("\n");
    }    
    free(avgvec[0]);
    free(avgvec[1]);
    free(avgvec);
    }
    if(strcmp(word, "wam") == 0 || strcmp(word, "gl") == 0 || strcmp(word, "ddg") == 0){
        if(strcmp(word, "wam") == 0){
            result = Createwam(p, m, (int)num/m);
        }  
        if(strcmp(word, "gl") == 0){
            result = Creategl(p, m, (int)num/m);
        } 
        if(strcmp(word, "ddg") == 0){
            result = Createddg(p, m, (int)num/m);
        }   
        n = (int)num/m ;
        free(p);   
        for( gg = 0; gg < n; gg++){
            printf("%.4f",result[gg*n]);
            for(k = 1;k < n;k++){
                printf(",%.4f",result[gg*n + k]); }
            printf("\n");
        }
        free(result);    
    }
    fclose(fp);
    return 1;

    
}

