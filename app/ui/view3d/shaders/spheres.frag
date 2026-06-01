#version 330 core
//
// Earthquake-sphere fragment shader.  Same Lambert + ambient model as the
// terrain and cylinder shaders so a sphere reads consistently against the
// surface it sits beneath.
//
in vec4 v_color;
in vec3 v_normal;

uniform vec3  u_sunDir;
uniform float u_ambient;

out vec4 frag;

void main() {
    vec3 N = normalize(v_normal);
    if (!gl_FrontFacing) N = -N;
    float ndotl = max(dot(N, normalize(u_sunDir)), 0.0);
    float lit = u_ambient + (1.0 - u_ambient) * ndotl;
    frag = vec4(v_color.rgb * lit, v_color.a);
}
