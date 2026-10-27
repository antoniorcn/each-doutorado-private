#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

int main() {
    // Inicializa a GLFW
    if (!glfwInit()) {
        std::cerr << "Falha ao inicializar a GLFW" << std::endl;
        return -1;
    }

    // Cria a janela
    GLFWwindow* window = glfwCreateWindow(800, 600, "Meu Primeiro Triangulo OpenGL", NULL, NULL);
    if (!window) {
        std::cerr << "Falha ao criar a janela GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Faz o contexto da janela ser o principal na thread atual
    glfwMakeContextCurrent(window);

    // Inicializa a GLEW (Gerenciador de Extensoes)
    if (glewInit() != GLEW_OK) {
        std::cerr << "Falha ao inicializar a GLEW" << std::endl;
        return -1;
    }

    // Loop principal do jogo/aplicacao
    while (!glfwWindowShouldClose(window)) {
        // Limpa a tela com uma cor cinza escuro
        glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Desenha um triangulo simples
        glBegin(GL_TRIANGLES);
            glColor3f(1.0f, 0.0f, 0.0f); // Vermelho
            glVertex2f(-0.5f, -0.5f);

            glColor3f(0.0f, 1.0f, 0.0f); // Verde
            glVertex2f(0.5f, -0.5f);

            glColor3f(0.0f, 0.0f, 1.0f); // Azul
            glVertex2f(0.0f, 0.5f);
        glEnd();

        // Troca os buffers de tela e trata eventos (cliques, teclado)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Finaliza as bibliotecas e fecha a janela
    glfwTerminate();
    return 0;
}
