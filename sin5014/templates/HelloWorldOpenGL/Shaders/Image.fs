uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform float iTime;
uniform vec4 iMouse;
uniform vec2 iResolution;
out vec4 C;
#define MAX_STEPS 100
#define MAX_DIST 100.
#define SURF_DIST .01
const float EPSILON = 0.0001;
mat2 Rot(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}

float smin( float a, float b, float k ) {
    float h = clamp( 0.5+0.5*(b-a)/k, 0., 1. );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
        vec3 ab = b-a;
    vec3 ap = p-a;

    float t = dot(ab, ap) / dot(ab, ab);
    t = clamp(t, 0., 1.);

    vec3 c = a + t*ab;

    return length(p-c)-r;
}

float sdCylinder(vec3 p, vec3 a, vec3 b, float r) {
        vec3 ab = b-a;
    vec3 ap = p-a;

    float t = dot(ab, ap) / dot(ab, ab);
    //t = clamp(t, 0., 1.);

    vec3 c = a + t*ab;

    float x = length(p-c)-r;
    float y = (abs(t-.5)-.5)*length(ab);
    float e = length(max(vec2(x, y), 0.));
    float i = min(max(x, y), 0.);

    return e+i;
}

float sdTorus(vec3 p, vec2 r) {
        float x = length(p.xz)-r.x;
    return length(vec2(x, p.y))-r.y;
}

float dBox(vec3 p, vec3 s) {
        return length(max(abs(p)-s, 0.));
}



float getDist(vec3 p)
{
    vec4 sphere = vec4(0.0,1.0,6.0,1.0);
    float sphereDist = length(p - sphere.xyz) - sphere.w;
    float planeDist = p.y;

    float cd = sdCapsule(p, vec3(3, .5, 6), vec3(3, 2.5, 6), .5);
    float td = sdTorus(p-vec3(0,.5,6), vec2(1.5, .4));
    float bd = dBox(p-vec3(-3.5, 1, 6), vec3(1,.75,1));
    float cyld = sdCylinder(p, vec3(0, .3, 3), vec3(3, .3, 5), .3);

    float d = min(cd, planeDist);
    d = min(d, td);
    d = min(d, bd);

    d = min(d, cyld);

    return d;

}


float RayMarching(vec3 ro, vec3 rd)
{
    float dO=0.0;
    vec3 p = vec3(0.0);
    float ds=MAX_DIST;
    int i=0;
    while((i<MAX_STEPS)&&(dO<MAX_DIST)&&(ds>SURF_DIST))
    {
        p = ro +dO*rd;
        ds = getDist(p);
        dO+=ds;
        i++;
    }
    return dO;
}

vec3 getNormal(vec3 p)
{
    float d = getDist(p);
    vec2 e = vec2(.01,0.0);
    vec3 n = d- vec3(getDist(p-e.xyy),getDist(p-e.yxy),getDist(p-e.yyx));
    return normalize(n);
}

/*//copa
    float r=0.09;
    r+= 0.037*cos(atan(q.x,q.y)*12.0-15.0*q.x+2.0*sin(iTime*2.0));
    r+=0.01*sin(atan(q.x,q.y)*80);
    col*=dist(q,r);

    //tronco
    r=0.02;
    r+=0.0025*cos(p.y*130.0);
    r+=exp(-40.0*p.y);
    col*=1.0 -(1.0-smoothstep(r,r+0.001,abs(q.x-0.1*sin(q.y*3.0))))*(1.0-smoothstep(0.0,0.1,q.y));

    // sol
    vec2 s = p - vec2(0.9,0.2);
    r = 0.05;
    col+=1-smoothstep(r,r+0.1,length(s)-r);*/


/**
 * Using the gradient of the SDF, estimate the normal on the surface at point p.
 */
vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        getDist(vec3(p.x + EPSILON, p.y, p.z)) - getDist(vec3(p.x - EPSILON, p.y, p.z)),
        getDist(vec3(p.x, p.y + EPSILON, p.z)) - getDist(vec3(p.x, p.y - EPSILON, p.z)),
        getDist(vec3(p.x, p.y, p.z  + EPSILON)) - getDist(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

/**
 * Lighting contribution of a single point light source via Phong illumination.
 *
 * The vec3 returned is the RGB color of the light's contribution.
 *
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 * lightPos: the position of the light
 * lightIntensity: color/intensity of the light
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity) {
    vec3 N = estimateNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));
    vec3 Resp=vec3(0.);
    float dotLN = dot(L, N);
    float dotRV = dot(R, V);

    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    }

    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        Resp = lightIntensity * (k_d * dotLN);
        //float d = RayMarching(p+N*SURF_DIST*2.0,L);
        //if(d <length(lightPos-p)) Resp*=0.1;
        return Resp;
    }
    Resp= lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));

    return Resp;
}

/**
 * Lighting via Phong illumination.
 *
 * The vec3 returned is the RGB color of that point after lighting is applied.
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
    const vec3 ambientLight = 0.5 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;

    vec3 light1Pos = vec3(4.0 * sin(iTime),
                          2.0,
                          4.0 * cos(iTime));
    vec3 light1Intensity = vec3(0.4, 0.4, 0.4);

    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light1Pos,
                                  light1Intensity);

    vec3 light2Pos = vec3(2.0 * sin(0.37 * iTime),
                          2.0 * cos(0.37 * iTime),
                          2.0);
    vec3 light2Intensity = vec3(0.4, 0.4, 0.4);

    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light2Pos,
                                  light2Intensity);

    return color;
}


float getLight(vec3 p)
{
    vec3 lightPos = vec3 (0.0,5.0,6.0);
    lightPos.xz+=vec2(cos(iTime),sin(iTime))*2.0;
    vec3 l = normalize(lightPos-p);
    vec3 n = getNormal(p);
    float diff= clamp(dot(n,l),0.0,1.0);
    float d = RayMarching(p+n*SURF_DIST*2.0,l);
    if(d <length(lightPos-p)) diff*=0.1;
    return diff;
}

void main()
{
    vec2 p = (gl_FragCoord.xy-0.5*iResolution)/iResolution.y;


    vec3 col = mix(vec3 (.01,0.1,0.1),vec3(0.8,0.8,1.0),-p.y);
    vec2 m = iMouse.z*(iMouse.xy/iResolution.xy);
    vec3 ro = vec3(0.0,2.0,-3.0);
    ro.yz *= Rot(-m.y+.4);
    ro.xz *= Rot(iTime*.2-m.x*6.2831);
    vec3 rd = normalize(vec3(p.x-0.1,p.y-0.2,1.0));
    float d=RayMarching(ro,rd);
    vec3 po =ro+d*rd;
    vec3 K_a = vec3(0.2, 0.2, 0.2);
        vec3 K_d = vec3(0.7, 0.2, 0.2);
        vec3 K_s = vec3(1.0, 1.0, 1.0);
        float shininess = 10.0;
        col = phongIllumination(K_a, K_d, K_s, shininess, po, ro);
    col = pow(col, vec3(.4545)); // gamma correction
    C = vec4( col, 1.0 );
}
