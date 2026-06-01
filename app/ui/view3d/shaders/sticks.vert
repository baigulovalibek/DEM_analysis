#version 330 core
//
// Earthquake-stick vertex shader.
//
// A single unit cylinder mesh is uploaded once (radius 1, height 1, axis +Z,
// base at z=0).  Per-instance attributes place that cylinder into the world
// frame as a stick spanning from the hypocenter (i_bottom) to the surface
// anchor (i_top), with the requested radius.  We apply u_zFactor here so the
// Z-exaggeration slider stretches the sticks together with the terrain mesh
// — without it, deep events would float above an exaggerated surface.
//
layout(location = 0) in vec3 a_unit;          // unit cylinder vertex
layout(location = 1) in vec3 a_unit_normal;   // unit cylinder normal

layout(location = 2) in vec3 i_bottom;        // hypocenter (per-instance)
layout(location = 3) in vec3 i_top;           // surface anchor (per-instance)
layout(location = 4) in float i_radius;       // stick half-width (per-instance)
layout(location = 5) in vec4 i_rgba;          // colour + opacity (per-instance)

uniform mat4  u_mvp;
uniform float u_zFactor;

out vec4 v_color;
out vec3 v_normal;

void main() {
    // Exaggerated Z for both endpoints — keeps the stick anchored to the
    // exaggerated terrain surface.
    float z_bot = i_bottom.z * u_zFactor;
    float z_top = i_top.z    * u_zFactor;

    // Place the unit cylinder.  XY: scale by radius around the stick axis
    // (i_bottom.xy == i_top.xy for vertical sticks).  Z: lerp between
    // bottom and top using the unit cylinder's z (0..1).
    vec3 world;
    world.xy = i_bottom.xy + a_unit.xy * i_radius;
    world.z  = mix(z_bot, z_top, a_unit.z);

    v_color  = i_rgba;
    v_normal = a_unit_normal;     // already unit-length from the CPU mesh

    gl_Position = u_mvp * vec4(world, 1.0);
}
