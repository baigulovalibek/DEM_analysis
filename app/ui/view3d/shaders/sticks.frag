#version 330 core
//
// Earthquake-stick fragment shader.
//
// Same Lambert + ambient pattern as the terrain shader so a stick lit by
// the same sun reads consistently against the surface it rises from.
// Front/back face flip mirrors the terrain shader: face culling is disabled
// scene-wide, so the inside of a cylinder seen end-on still needs to be lit.
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
