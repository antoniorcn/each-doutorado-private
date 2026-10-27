uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform float iTime;
uniform vec4 iMouse;
uniform vec2 iResolution;
uniform int iFrame;
out vec4 C;

#define PI 3.1459

/*  Install  Istructions

sudo apt-get install g++ cmake git
 sudo apt-get install libsoil-dev libglm-dev libassimp-dev libglew-dev libglfw3-dev libxinerama-dev libxcursor-dev
libxi-dev libfreetype-dev libgl1-mesa-dev xorg-dev

git clone https://github.com/JoeyDeVries/LearnOpenGL.git*/


float dist(vec2 p, float r)
{
    float d = length(p)-r;
    return smoothstep(0.0,0.0001,d);
}


void main ()
{
    vec2 p =gl_FragCoord.xy/iResolution.xy;
    float ra = iResolution.x/iResolution.y;
    p.x*=ra;
    vec3 c1 = vec3(1.0,0.4,0.1);
    vec3 c2 = vec3(1.0,0.8,0.4);
    vec3 col = mix(c1,c2,p.y);
    float r=0.1;
    vec2 q = p-vec2(0.5,0.7);

    vec2 q2 = p-vec2(0.8, 0.2-0.45*(0.5+0.5*sin(0.05*(iTime*2.0*PI))));

    r+=0.065*cos(atan(q.x/q.y)*12.0- 20.0*q.x +sin(iTime*PI));
    r+=0.02*sin(atan(q.x/q.y)*100.0);
    col*=dist(q,r);

    r = 0.25;
    r+=0.03*sin(atan(q2.x/q2.y)*200.0 - sin(iTime*PI));
    // r+=0.03*sin(atan(q2.x/q2.y)* iTime*0.5);
    float cor_sol = 1 - dist(q2,r);
    col+=vec3(cor_sol, cor_sol, 0);


    r=0.02;
    r+=0.0025*cos(q.y*120.0);
    r+=exp(-40.0*p.y);
    col*=1.0-(1.0-smoothstep(r,r+0.001,abs(q.x-0.1*sin(q.y*4.0))))*(1.0-smoothstep(0.0,0.1,q.y));


     C=vec4(col,1.0);

}
