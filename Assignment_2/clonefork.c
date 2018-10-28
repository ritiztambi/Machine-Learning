#define _GNU_SOURCE
#include <sched.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
int variable, fd;

int child_func() {
   variable = 8;
   close(fd);
   return(0);
}

int main(int argc, char *argv[]) {
   void **child_stack;
   char ch;

   variable = -1;
   fd = open("csci.log", O_RDONLY);
   child_stack = (void **) malloc(16384);
   printf("The variable was %d\n", variable);

   clone(child_func, child_stack+16384,0, NULL);
   sleep(1);

   printf("The variable is now %d\n", variable);
   if (read(fd, &ch, 1) < 1) {
      perror("File Read Error");
      return(1);
   }
   printf("Successfull read from the file\n");
   return(0);
}
