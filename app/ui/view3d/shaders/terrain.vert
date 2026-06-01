#version 330 core
//
// Terrain vertex shader.
//
// Vertices are uploaded in world-space metres with z already representing
// elevation in metres.  We apply the vertical-exaggeration factor here so the
// Z-slider re-shades the scene without a CPU rebuild.  Normals are scaled by
// the inverse Z-factor so lighting matches the apparent slope after
// exaggeration.
//
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4  u_mvp;
uniform mat4  u_model;          // currently identity, kept for future xforms
uniform float u_zFactor;

out vec3 v_normal;
out vec2 v_uv;
out vec3 v_worldPos;
out float v_height;

void main() {
    vec3 p = vec3(a_pos.xy, a_pos.z * u_zFactor);

    // Re-tilt normals so they match the exaggerated surface.  At z_factor=1
    // this is a no-op; at z_factor>1 normals rotate toward horizontal,
    // which is what we want — exaggerated slopes look more lit.
    vec3 n = vec3(a_normal.xy, a_normal.z / max(u_zFactor, 0.001));
    v_normal   = normalize(n);
    v_uv       = a_uv;
    v_worldPos = p;
    v_height   = a_pos.z;        // un-exaggerated elevation, used by frag
    gl_Position = u_mvp * vec4(p, 1.0);
}
