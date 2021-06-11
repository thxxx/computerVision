all : project3

program1 : prject3.c
	gcc -o $@ $<

clean :
	rm -f project3