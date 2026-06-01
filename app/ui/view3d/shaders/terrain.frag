#version 330 core
//
// Terrain fragment shader.
//
// Lambert + ambient using the drape texture as albedo.  Background colour is
// used to fade out far geometry slightly so it doesn't pop against the sky;
// the effect is subtle (5–10% blend at the far plane).
//
in vec3 v_normal;
in vec2 v_uv;
in vec3 v_worldPos;
in float v_height;

uniform sampler2D u_drape;
uniform vec3  u_sunDir;          // pre-normalised, points TOWARD sun
uniform float u_ambient;         // 0..1
uniform vec3  u_background;
uniform int   u_hasTexture;      // 1 → sample u_drape, 0 → procedural ramp
uniform float u_zMin;
uniform float u_zMax;

out vec4 frag;

vec3 procedural_terrain_color(float h) {
    // Fallback colour ramp used when no drape texture is loaded.
    // Matches "terrain" cmap reasonably well: low→blue, mid→green, high→white.
    float t = clamp((h - u_zMin) / max(u_zMax - u_zMin, 1.0), 0.0, 1.0);
    vec3 low  = vec3(0.20, 0.40, 0.55);
    vec3 mid  = vec3(0.40, 0.55, 0.30);
    vec3 high = vec3(0.85, 0.78, 0.65);
    vec3 snow = vec3(0.98, 0.98, 0.98);
    vec3 c;
    if      (t < 0.33) c = mix(low,  mid,  t / 0.33);
    else if (t < 0.75) c = mix(mid,  high, (t - 0.33) / 0.42);
    else               c = mix(high, snow, (t - 0.75) / 0.25);
    return c;
}

void main() {
    vec4 albedo;
    if (u_hasTexture == 1) {
        albedo = texture(u_drape, v_uv);
        if (albedo.a < 0.01) {
            // Texture says nodata — fall back to procedural so we still see
            // the terrain mesh.
            albedo = vec4(procedural_terrain_color(v_height), 1.0);
        }
    } else {
        albedo = vec4(procedural_terrain_color(v_height), 1.0);
    }

    vec3 N = normalize(v_normal);
    // Back-face culling is disabled (terrain is an open surface).  Flip the
    // normal on back-facing fragments so the underside is lit too — without
    // this, fragments seen from below were rendering pitch black at any
    // tilt > 90°.
    if (!gl_FrontFacing) N = -N;
    float ndotl = max(dot(N, normalize(u_sunDir)), 0.0);
    vec3 lit = albedo.rgb * (u_ambient + (1.0 - u_ambient) * ndotl);

    frag = vec4(lit, 1.0);
}
