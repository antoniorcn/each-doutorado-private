CXX = g++
CC = gcc

CXXFLAGS = -std=c++17 -Wall -Wextra
CFLAGS = -Wall -Wextra

INCLUDES = -Iincludes -I.

LIBS = -lglfw3 -lopengl32 -lgdi32

TARGET = programa.exe

all: glad.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) Introduction.cpp glad.o -o $(TARGET) $(LIBS)

glad.o: glad.c
	$(CC) $(CFLAGS) $(INCLUDES) -c glad.c -o glad.o

clean:
	-del /Q $(TARGET) glad.o
