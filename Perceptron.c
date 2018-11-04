#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define ROWS 20
#define COLS 8
#define TESTROWS 4

#define epoch 10  //Number of epochs for perceptron training

#define n 0.1   //Learning rate

typedef struct _tab {
   int *x;
   int y;
} tab;

double sigmoid(double val) {   //Sigmoid function
   return (1.0/(1.0+exp(-val)));
}

double Perceptron(int *instance, double w[], double w0) {   //Perceptron, returns sigmoid(Summation wi*xi)

   double sum = w0;
   for(int i=0; i<COLS; i++) sum += instance[i]*w[i];
  
   return sigmoid(sum);

}

void Deltalearn(tab *table, double w[], double w0) {   //Delta learning algorithm

   for(int k=0; k<ROWS; k++) {

      for(int i=0; i<epoch; i++) {

         w0 += n*(table[k].y-Perceptron(table[k].x, w, w0));  //Updates w0 <- w0 + n*(t-o)

         for(int j=0; j<COLS; j++) {

            w[j] += n*(table[k].y-Perceptron(table[k].x, w, w0))*table[k].x[j];   //Updates wi <- wi + n*(t-o)*xi

         }

      }

   }

}

int classify(int *instance, double w[], double w0) {   //Classifies value returned from perceptron to either 0 or 1

   return (Perceptron(instance, w, w0) >= 0.5);

}

int main()

{

   FILE *fp;
   tab *table = (tab *) malloc(ROWS*sizeof(tab));
   int **instance = (int **) malloc(TESTROWS*sizeof(int *));
  
   double *w = (double *) calloc(COLS, sizeof(double));   //Initializes weights wi to 0
   double w0 = 0;   //Initializes offset w0 to 0

   fp = fopen("data6.csv", "r");

   if(fp == NULL) {
      printf("data6.csv not found\n");
      return 1;
   }

   for(int i=0; i<ROWS; i++) {
   
      table[i].x = (int *) malloc(COLS*sizeof(int));

      for(int j=0; j<COLS; j++) {

         fscanf(fp, " %d,", &table[i].x[j]);

      }

      fscanf(fp, " %d,", &table[i].y);

   }

   fclose(fp);

   fp = fopen("test6.csv", "r");

   if(fp == NULL) {
      printf("test6.csv not found\n");
      return 1;
   }

   for(int i=0; i<TESTROWS; i++) {
 
      instance[i] = (int *) malloc(COLS*sizeof(int));

      for(int j=0; j<COLS; j++) {

         fscanf(fp, " %d,", &instance[i][j]);

      }

   }

   fclose(fp);

   Deltalearn(table, w, w0);   //Trains perceptron

   fp = fopen("Perceptron.out", "w");

   for(int i=0; i<TESTROWS; i++) {

      printf("%d ", classify(instance[i], w, w0));
      fprintf(fp, "%d ", classify(instance[i], w, w0));

   }

   printf("\n");

   fclose(fp);

   return 0;

}
