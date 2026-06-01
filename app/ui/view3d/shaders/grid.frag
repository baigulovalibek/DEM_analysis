#version 330 core
//
// Coordinate-grid fragment shader — just emit the interpolated colour.
//
in vec4 v_color;
out vec4 frag;

void main() {
    frag = v_color;
}
