#include <iostream>
#include <cuda.h>
using namespace std;

__global__ void MatrixMulkernel(int *c, const int* a,const int* b,int width){
    int sum=0;
    int x,y,i;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    x = blockIdx.x * blockDim.x + threadIdx.x;

    for(int k=0; k<width; k++){
       sum += a[y* width + k] * b[k* width +x];
    }
    i = y * width + x;
    c[i] = sum;
   
}

void PrintMatrix(int a[][16], int size){
    for(int i=0; i<size; i++){
        printf("|");
        for(int j=0; j<size; j++){
            printf("%6d",a[i][j]);
            if(j==7) printf("|      |");
        }
        
        printf("|\n");
        if(i==7) printf("\n");
    }
}

int main(){
    const int WIDTH = 16;
    const int TILE_WIDTH = 2;
    int a[WIDTH][WIDTH];
    int b[WIDTH][WIDTH];
    int c[WIDTH][WIDTH] = {0,};

    for(int y=0; y<WIDTH; y++){
        for(int x=0; x<WIDTH; x++){
            a[y][x] = y+x;
            b[y][x] = y*10+x; 
        }
    }

    int *dev_a, *dev_b, *dev_c=0;
    cudaMalloc((void**)&dev_a, WIDTH*WIDTH*sizeof(int));
    cudaMalloc((void**)&dev_b, WIDTH*WIDTH*sizeof(int));
    cudaMalloc((void**)&dev_c, WIDTH*WIDTH*sizeof(int));

    cudaMemcpy(dev_a, a, WIDTH*WIDTH*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, WIDTH*WIDTH*sizeof(int), cudaMemcpyHostToDevice);
    dim3 dimGrid(WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH,1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH,1);

    MatrixMulkernel<<<dimGrid,dimBlock>>>(dev_c,dev_a,dev_b,WIDTH);
    cudaDeviceSynchronize();

    cudaMemcpy(c,dev_c,WIDTH*WIDTH*sizeof(int),cudaMemcpyDeviceToHost);
    printf("<Matrix a> \n");
    PrintMatrix(a,WIDTH);printf("\n\n");
    printf("<Matrix b> \n");
    PrintMatrix(b,WIDTH);printf("\n\n");
    printf("<Matrix c> \n");
    PrintMatrix(c,WIDTH);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
