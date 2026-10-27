# Nome do compilador
CXX = g++

# Flags de compilação (ativa avisos de erros comuns)
CXXFLAGS = -Wall -Wextra -std=c++17

# Bibliotecas do OpenGL que precisamos linkar
LIBS = -lglew32 -lglfw3 -lopengl32 -lgdi32

# Nome do arquivo executável final
TARGET = programa.exe

# Arquivos fontes (.cpp)
SRCS = main.cpp

# Regra padrão executada ao rodar apenas 'mingw32-make'
all: $(TARGET)

# Regra para compilar o executável
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

# Regra para limpar os arquivos gerados
clean:
	del $(TARGET)
