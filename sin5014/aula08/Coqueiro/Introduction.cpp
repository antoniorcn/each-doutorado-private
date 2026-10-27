#include <glad/glad.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include "Shader.h"
#include "stb_image/stb_image.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <vector>


#define PI 3.1459

double lastTime,lt,fps;
int nbFrames = 0;
char title_string[50];
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window,double *mouse, bool * shouldDraw, bool * press);
void DisplayFramebufferTexture(unsigned int textureID, Shader *program,unsigned int VAO,glm::vec2 R);
const unsigned int SCR_WIDTH = 1024;
const unsigned int SCR_HEIGHT = 768;


float vertices[] = {
    1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
    1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
    -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
    -1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left
};

float TVvertices[] = {
    1.0f,  1.0f,   1.0f, 1.0f, // top right
    1.0f, -1.0f,   1.0f, 0.0f, // bottom right
    -1.0f, -1.0f,   0.0f, 0.0f, // bottom left
    -1.0f,  1.0f,   0.0f, 1.0f  // top left
};

unsigned int indices[] = {
    0,1,3,
    1,2,3
};

float dist(glm::vec2 p, float r)
{
    float d = length(p)-r;
    return glm::smoothstep(0.0f, 0.0001f, d);
}



glm::vec4 desenhar(glm::vec4 gl_FragCoord, glm::vec2 iResolution, float iTime) { 
    glm::vec2 p = glm::vec2(gl_FragCoord.x, gl_FragCoord.y) / glm::vec2(iResolution.x, iResolution.y);
    float ra = iResolution.x/iResolution.y;
    p.x*=ra;
    glm::vec3 c1 = glm::vec3(1.0,0.4,0.1);
    glm::vec3 c2 = glm::vec3(1.0,0.8,0.4);
    glm::vec3 col = mix(c1,c2,p.y);
    float r=0.1;
    glm::vec2 q = p-glm::vec2(0.5,0.7);

    glm::vec2 q2 = p-glm::vec2(0.8, 0.2-0.45*(0.5+0.5*sin(0.05*(iTime*2.0*PI))));

    r+=0.065*cos(atan(q.x/q.y)*12.0- 20.0*q.x +sin(iTime*PI));
    r+=0.02*sin(atan(q.x/q.y)*100.0);
    col*=dist(q,r);

    r = 0.25;
    r+=0.03*sin(atan(q2.x/q2.y)*200.0 - sin(iTime*PI));
    // r+=0.03*sin(atan(q2.x/q2.y)* iTime*0.5);
    float cor_sol = 1 - dist(q2,r);
    col+=glm::vec3(cor_sol, cor_sol, 0);


    r=0.02;
    r+=0.0025*cos(q.y*120.0);
    r+=exp(-40.0*p.y);
    col*=1.0-(1.0-glm::smoothstep((float)r, (float)r + 0.001f, (float)abs(q.x-0.1f*sin(q.y*4.0f))))*(1.0-glm::smoothstep(0.0f, 0.1f, q.y));


     return glm::vec4(col,1.0);
}


int main()
{
    
    if (__cplusplus == 201703L)
        std::cout << "C++17" << std::endl;
    else if (__cplusplus == 201402L)
        std::cout << "C++14" << std::endl;
    else if (__cplusplus == 201103L)
        std::cout << "C++11" << std::endl;
    else if (__cplusplus == 199711L)
        std::cout << "C++98" << std::endl;
    else
        std::cout << "pre-standard C++" << std::endl;


    std::filesystem::path path = std::filesystem::current_path();
    const std::string pathString = path.string();

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Introduction", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    std::string parentPath = path.string();

    const std::string vertexShader = "/Shaders/VertexShader.vs";
    const std::string vertexPath = parentPath + vertexShader;
    const char * vertexShaderPath = vertexPath.c_str();

    const std::string TVvertexShader = "/Shaders/TextureViewer.vs.vert";
    const std::string TVvertexPath = parentPath + TVvertexShader;
    const char * TVvertexShaderPath = TVvertexPath.c_str();


    const std::string fragmentShader = "/Shaders/Image.fs.frag";
    const std::string fragmentPath = parentPath + fragmentShader;
    const char * fragmentShaderPath = fragmentPath.c_str();
   


    const std::string fragmentShaderA = "/Shaders/BufferA.fs";
    const std::string fragmentPathA = parentPath + fragmentShaderA;
    const char * fragmentShaderAPath = fragmentPathA.c_str();


    const std::string fragmentShaderB = "/Shaders/BufferB.fs";
    const std::string fragmentPathB = parentPath + fragmentShaderB;
    const char * fragmentShaderBPath = fragmentPathB.c_str();

    const std::string fragmentShaderC = "/Shaders/BufferC.fs";
    const std::string fragmentPathC = parentPath + fragmentShaderC;
    const char * fragmentShaderCPath = fragmentPathC.c_str();

    const std::string fragmentShaderD = "/Shaders/BufferD.fs";
    const std::string fragmentPathD = parentPath + fragmentShaderD;
    const char * fragmentShaderDPath = fragmentPathD.c_str();

    const std::string TVShaderString = "/Shaders/TextureViewer.fs.frag";
    const std::string TVShaderPathST = parentPath + TVShaderString;
    const char * TVShaderPath = TVShaderPathST.c_str();

    const std::string cString = parentPath+ "/Shaders/common.incl";
    const char * commonPath = cString.c_str();
    // Shader Imageprogram(commonPath, vertexShaderPath, fragmentShaderPath);
    Shader Imageprogram(commonPath, vertexShaderPath, fragmentShaderPath);
    
    ////////////////////
    // OPENGL BÁSICOS //
    ////////////////////
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER,VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STREAM_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices),indices, GL_STREAM_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float),(void *)0 );
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float),(void *) (3*sizeof(float)) );
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float),(void *) (6*sizeof(float)) );
    glEnableVertexAttribArray(2);


    ////////////////////
    // OPENGL BÁSICOS //
    ////////////////////
    unsigned int TV_VBO, TV_VAO, TV_EBO;
    glGenVertexArrays(1, &TV_VAO);
    glGenBuffers(1, &TV_VBO);
    glGenBuffers(1, &TV_EBO);
    glBindVertexArray(TV_VAO);

    glBindBuffer(GL_ARRAY_BUFFER,TV_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(TVvertices), TVvertices, GL_STREAM_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, TV_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices),indices, GL_STREAM_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float),(void *)0 );
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float),(void *) (2*sizeof(float)) );
    glEnableVertexAttribArray(1);


    ////////////////////
    //  FRAMEBUFFER   //
    ////////////////////
     

    //Criando o framebuffer
    unsigned int FBO_0;
    glGenFramebuffers(1, &FBO_0);

    //Ligando o framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER,FBO_0);

    //Para utilizar o framebuffer, é necessário ligar a ele uma textura
    unsigned int iChannel_0;

    glGenTextures(1, &iChannel_0);
    glBindTexture(GL_TEXTURE_2D, iChannel_0);
     
    //Gerando uma textura vazia para ligar ao framebuffer
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);

    //LIGANDO A TEXTURA AO FRAMBUFFER

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, iChannel_0, 0 );

    // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
        unsigned int rbo_0;
        glGenRenderbuffers(1, &rbo_0);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_0);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT); // use a single renderbuffer object for both a depth AND stencil buffer.
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo_0); // now actually attach it
        // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
     //glBindFramebuffer(GL_FRAMEBUFFER, 0);


    //Criando o framebuffer
    unsigned int FBO_1;
    glGenFramebuffers(1, &FBO_1);
    //Ligando o framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER,FBO_1);


    unsigned int iChannel_1;

    glGenTextures(1, &iChannel_1);
    glBindTexture(GL_TEXTURE_2D, iChannel_1);

    //Gerando uma textura vazia para ligar ao framebuffer
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);

    //LIGANDO A TEXTURA AO FRAMBUFFER

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, iChannel_1, 0);

    // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
    unsigned int rbo_1;
    glGenRenderbuffers(1, &rbo_1);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_1);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT); // use a single renderbuffer object for both a depth AND stencil buffer.
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo_1); // now actually attach it
    // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
         std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
    //glBindFramebuffer(GL_FRAMEBUFFER, 0);

    //Criando o framebuffer
    unsigned int FBO_2;
    glGenFramebuffers(1, &FBO_2);
    //Ligando o framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER,FBO_2);

    unsigned int iChannel_2;

    glGenTextures(1, &iChannel_2);
    glBindTexture(GL_TEXTURE_2D, iChannel_2);

    //Gerando uma textura vazia para ligar ao framebuffer
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);

    //LIGANDO A TEXTURA AO FRAMBUFFER
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, iChannel_2, 0 );

    // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
    unsigned int rbo_2;
    glGenRenderbuffers(1, &rbo_2);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_2);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT); // use a single renderbuffer object for both a depth AND stencil buffer.
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo_2); // now actually attach it
    // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now


    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
         std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
    //glBindFramebuffer(GL_FRAMEBUFFER, 0);


    //Criando o framebuffer
    unsigned int FBO_3;
    glGenFramebuffers(1, &FBO_3);
    
    //Ligando o framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER,FBO_3);

    unsigned int iChannel_3;

    glGenTextures(1, &iChannel_3);
    glBindTexture(GL_TEXTURE_2D, iChannel_3);

    //Gerando uma textura vazia para ligar ao framebuffer
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);

    //LIGANDO A TEXTURA AO FRAMBUFFER
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, iChannel_3, 0 );

    // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
        unsigned int rbo_3;
        glGenRenderbuffers(1, &rbo_3);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_3);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT); // use a single renderbuffer object for both a depth AND stencil buffer.
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo_3); // now actually attach it
        // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now


    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
         std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;

    double * mouse = (double *) malloc(sizeof(double)*4);

    glm::vec2 resolution(SCR_WIDTH, SCR_HEIGHT);
    //Shaderprogram2.setVec2("iResolution",resolution);
    Imageprogram.use();
    Imageprogram.setVec2("iResolution", resolution);

    bool shouldDraw = false;
    bool press = false;
    float currentTime;
    int frame=0;

    // Antes do while principal:
    std::vector<unsigned char> pixels(
        static_cast<std::size_t>(SCR_WIDTH) * SCR_HEIGHT * 4
    );

    ////////////////////
    //  RENDER LOOP   //
    ////////////////////
    while(!glfwWindowShouldClose(window)){
        currentTime = static_cast<float>(glfwGetTime());
        //Renderizando para o framebuffer criado

        processInput(window, mouse, &shouldDraw, & press);

        if(shouldDraw) {

            //Renderizando para a tela
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            auto paraByte = [](float valor) {
                valor = std::clamp(valor, 0.0f, 1.0f);
                return static_cast<unsigned char>(std::lround(valor * 255.0f));
            };

            for (int y = 0; y < SCR_HEIGHT; ++y) {
                for (int x = 0; x < SCR_WIDTH; ++x) {
                    glm::vec4 fragCoord(
                        static_cast<float>(x) + 0.5f,
                        static_cast<float>(y) + 0.5f,
                        0.0f,
                        1.0f
                    );

                    glm::vec4 cor = desenhar(fragCoord, resolution, currentTime);
                    const std::size_t indice =
                                (static_cast<std::size_t>(y) * SCR_WIDTH + x) * 4;

                    pixels[indice + 0] = paraByte(cor.r);
                    pixels[indice + 1] = paraByte(cor.g);
                    pixels[indice + 2] = paraByte(cor.b);
                    pixels[indice + 3] = paraByte(cor.a);
                            // Aqui, grave "cor" no buffer de pixels da imagem.
                }
            }

            
        
            // Imageprogram.use();

            glBindVertexArray(VAO);
            // Imageprogram.setVec2("iResolution",resolution) ;
       
            // Imageprogram.setFloat("iTime",currentTime);
            // Imageprogram.setInt("iFrame",frame);
            // Imageprogram.setVec4("iMouse", mouse);
        
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT,0);
        
            double CurrentTime = glfwGetTime();
            double past = CurrentTime - lt;
            if(past>0.1) {
                fps = (float)nbFrames/past;
                sprintf(title_string, "Introduction - FPS = %.i ", (int)fps);
                glfwSetWindowTitle(window, title_string);
                lt+=past;
                nbFrames = 0;
            }
            nbFrames++;
            frame++;
        }
        glfwSwapBuffers(window);
        glfwPollEvents();

    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteFramebuffers(1, &FBO_0);
    glDeleteFramebuffers(1, &FBO_1);
    glDeleteFramebuffers(1, &FBO_2);
    glDeleteFramebuffers(1, &FBO_3);
    glDeleteRenderbuffers(1, &rbo_0);
    glDeleteRenderbuffers(1, &rbo_1);
    glDeleteRenderbuffers(1, &rbo_2);
    glDeleteRenderbuffers(1, &rbo_3);

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow *window,double *mouse, bool * shouldDraw, bool * press)
{
    //Caso aperte 'esc', indica que a janela deve ser fechada
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

    //Alocando o vetor que contém o mouse
    //double * mouse = (double *) malloc(sizeof(double)*4);

    //Encontrando a posição do mouse
    glfwGetCursorPos(window,&mouse[0], &mouse[1]);
    mouse[2] = 0.0;
    mouse[3] = 0.0;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) mouse[2] = 1.0;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) mouse[3] = 1.0;

    //Normalizando as coordenadas do cursor
    int h,w;
    glfwGetFramebufferSize(window, &w, &h);
    //mouse[1] = h-mouse[1];
    //mouse[0] = ((mouse[0]*2.0)/w) -1.0;
    //mouse[1] = ((mouse[1]*2.0)/h) -1.0;
    mouse[1]=h-mouse[1];
    //return mouse;

    if(glfwGetKey(window, 32) == GLFW_PRESS && !*press) {
        *shouldDraw = !*shouldDraw;
        lt = glfwGetTime();
        * press = true;
    }
    if(glfwGetKey(window, 32) == GLFW_RELEASE) * press = false;
}

void DisplayFramebufferTexture(unsigned int textureID,Shader *program, unsigned int VAO,glm::vec2 R)
{
       program->use();
           program->setSampler("fboAttachment",0);
           program->setVec2("iResolution",R);
           glBindVertexArray(VAO);

           glActiveTexture(GL_TEXTURE0);
           glBindTexture(GL_TEXTURE_2D, textureID);

           glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT,0);
           glBindVertexArray(0);
       glUseProgram(0);
}

//call: chamada quando um determinado evento acontece
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
    printf("%i:%i\n", width,height);
}
