#version 330 core
//
// Coordinate-grid vertex shader.  Unlit — colour comes from the per-vertex
// attribute the CPU side baked in.  Z is exaggerated together with terrain
// and earthquake geometry so the grid box stays in lockstep with what the
// user sees.
//
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec4 a_color;

uniform mat4  u_mvp;
uniform float u_zFactor;

out vec4 v_color;

void main() {
    vec3 world = vec3(a_pos.x, a_pos.y, a_pos.z * u_zFactor);
    v_color = a_color;
    gl_Position = u_mvp * vec4(world, 1.0);
}
