#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<cuda_runtime.h>

// Function to calculate the number of digits in a number
__device__ int numDigits(long long n) {
	int count = 0;
	while (n != 0) {
		n /= 10;
		count++;
	}
	return count;
}

__device__ int customMax(int a, int b) {
    return (a > b) ? a : b;
}

__global__ void multiplication(long long *d_a, long long *d_b, long long *d_c, int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<len){
       int x=d_a[tid];
        int y=d_b[tid];
        if (x < 10 || y < 10) {
			d_c[tid] = x * y;
		}else{
			// Calculate the number of digits in the two numbers and divide by 2
			int n = customMax(numDigits(x), numDigits(y));
			int n2 = (n / 2);

			// Split the numbers into two parts

			long long x_h = x / (long long)pow(10, n2);
			long long x_l = x % (long long)pow(10, n2);
			long long y_h = y / (long long)pow(10, n2);
			long long y_l = y % (long long)pow(10, n2);

			// Recursively calculate the three products
			long long high_prod = x_h * y_h;
			long long low_prod = x_l * y_l;
			long long inter_prod = ((x_h + x_l) * (y_h + y_l));
			long long subtract = inter_prod - high_prod - low_prod;

			// Calculate and return the final result
			d_c[tid] = (high_prod * (long long)pow(10, 2 * n2)) + (subtract * (long long)pow(10, n2)) + low_prod;
		}
    }
}

int main(int argc, char const *argv[]){

    long long *d_a, *d_b, *d_c;
    long long C[100000];
    float import_time;
    float exe_time;
    float export_time;

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1, 0);

    FILE *file1 = fopen("X_100000.txt", "r");
	if (file1 == NULL) {
        printf("Failed to open the file for reading.\n");
        return 1;
    }
    long long A[100000];
    int num_elements_A = 0;

    // Read integers from the file and store them in an array
    while (fscanf(file1, "%lld", &A[num_elements_A]) != EOF) {
        num_elements_A++;
    }

    int num_of_elements = num_elements_A;

    FILE *file2 = fopen("Y_100000.txt", "r");
	if (file2 == NULL) {
        printf("Failed to open the file for reading.\n");
        return 1;
    }
    long long B[100000];
    int num_elements_B = 0;

    // Read integers from the file and store them in an array
    while (fscanf(file2, "%lld", &B[num_elements_B]) != EOF) {
        num_elements_B++;
    }

    cudaEventRecord(stop1, 0);
    cudaEventElapsedTime(&import_time, start1, stop1);

    cudaMalloc((void **)&d_a, num_of_elements*sizeof(long long int));
    cudaMalloc((void **)&d_b, num_of_elements*sizeof(long long int));
    cudaMalloc((void **)&d_c, num_of_elements*sizeof(long long int));

    cudaMemcpy(d_a, A, num_of_elements*sizeof(long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, num_of_elements*sizeof(long long int), cudaMemcpyHostToDevice);

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2, 0);

    int blockSize = 256;  // You can adjust this based on your requirements
    int numBlocks = (num_of_elements + blockSize - 1) / blockSize;
    multiplication<<<numBlocks, blockSize>>>(d_a, d_b, d_c, num_of_elements);

    cudaMemcpy(C, d_c, num_of_elements*sizeof(long long int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop2, 0);
    cudaEventElapsedTime(&exe_time, start2, stop2);

    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    cudaEventRecord(start3, 0);

    FILE *file3 = fopen("cudaproduct_100000.txt", "w");
    if (file3 == NULL) {
        printf("Failed to open the file for writing.\n");
        return 1;
    }
    for (int i = 0; i < num_of_elements; i++) {
        fprintf(file3, "%llu\n", C[i]);
    }

    cudaEventRecord(stop3, 0);
    cudaEventElapsedTime(&export_time, start3, stop3);

    printf("Time taken for importing dataset is : %fms.\n", import_time);
    printf("Time taken for Execution is : %fms.\n", exe_time);
    printf("Time taken for exporting dataset is : %fms.\n", export_time);
    printf("Product values are available in file cudaproduct_100000.txt\n");
	fclose(file1);
	fclose(file2);
	fclose(file3);
	return 0;
}