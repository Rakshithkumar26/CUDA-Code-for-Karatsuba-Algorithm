#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));
    FILE *file = fopen("Y_100000.txt", "w");
    if (file == NULL) {
        printf("Failed to open the file for writing.\n");
        return 1;
    }

    for (int i = 0; i < 100000; i++) {
        int random_number = rand() % 90000000 + 10000000;
        fprintf(file, "%d\n", random_number);
    }

    fclose(file);
    printf("Random numbers have been written to 'Y_100000.txt'\n");
    return 0;
}
