#version 330 core
//
// Earthquake-sphere vertex shader.
//
// A single unit sphere (radius 1, centred at origin) is uploaded once; per
// instance we place a copy at the event's hypocenter and scale it by the
// per-instance radius.  The sphere's *centre* Z is exaggerated by u_zFactor
// so it tracks the exaggerated terrain, but the radius is not — keeping
// spheres spherical instead of stretching them into ellipsoids when the
// user dials Z×.
//
layout(location = 0) in vec3 a_unit;          // unit sphere vertex (|a|=1)
layout(location = 1) in vec3 a_unit_normal;   // unit sphere normal (== a_unit)

// Per-instance attributes mirror the cylinder shader's so the GL widget
// can share the same instance VBOs between the two shapes:
//   i_bottom — sphere centre (x, y, z)
//   i_top    — unused for spheres
//   i_radius — sphere radius (metres)
//   i_rgba   — colour + opacity
layout(location = 2) in vec3 i_bottom;
layout(location = 3) in vec3 i_top;
layout(location = 4) in float i_radius;
layout(location = 5) in vec4 i_rgba;

uniform mat4  u_mvp;
uniform float u_zFactor;

out vec4 v_color;
out vec3 v_normal;

void main() {
    vec3 centre = vec3(i_bottom.x, i_bottom.y, i_bottom.z * u_zFactor);
    vec3 world = centre + a_unit * i_radius;

    v_color = i_rgba;
    v_normal = a_unit_normal;     // radial outward, already unit-length

    gl_Position = u_mvp * vec4(world, 1.0);
}
