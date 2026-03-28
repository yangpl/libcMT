CC = mpicc
CFLAGS = -O3 -g -Wall

HDF5_INC := /usr/include/hdf5/serial #$(shell pkg-config --cflags hdf5)
HDF5_LIB := /usr/lib/x86_64-linux-gnu/hdf5/serial #$(shell pkg-config --libs hdf5)

BIN = ../bin
TARGET = $(BIN)/libcMT
LIB = -lm -L$(HDF5_LIB) -lhdf5
INC = -I../include -I$(HDF5_INC)
SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)



all: clean $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LIB)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(INC)

clean:
	find . -name "*.o" -exec rm {} \;
	find . -name "*.c%" -exec rm {} \;
	find . -name "*.bck" -exec rm {} \;
	find . -name "*~" -exec rm {} \;
	find . -name "\\#*" -exec rm {} \;
	rm -f $(TARGET)
