// Box-downsample the (already transformed) input image to a small linear
// proxy for the film simulation's metering and light-scattering field.

struct ProxyParams {
    factor: u32,
    is_raw: u32,
    out_width: u32,
    out_height: u32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: ProxyParams;

fn srgb_to_linear_proxy(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.out_width || id.y >= params.out_height) {
        return;
    }
    let dims = textureDimensions(src);
    let f = params.factor;
    let x0 = id.x * f;
    let y0 = id.y * f;
    var sum = vec3<f32>(0.0);
    var n = 0.0;
    for (var dy = 0u; dy < f; dy++) {
        for (var dx = 0u; dx < f; dx++) {
            let x = x0 + dx;
            let y = y0 + dy;
            if (x < dims.x && y < dims.y) {
                var c = max(textureLoad(src, vec2<u32>(x, y), 0).rgb, vec3<f32>(0.0));
                if (params.is_raw == 0u) {
                    c = srgb_to_linear_proxy(c);
                }
                sum += c;
                n += 1.0;
            }
        }
    }
    textureStore(dst, id.xy, vec4<f32>(sum / max(n, 1.0), 1.0));
}
