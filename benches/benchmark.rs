use criterion::{black_box, criterion_group, criterion_main, Criterion};

use fast_srgb8::*;

fn prepare_data(array_len: usize, count: usize) -> Vec<f32> {
    // total samples = array_len x count + 1
    let total = array_len * count;
    let step = 1.0_f32 / total as f32;
    (0..=array_len * count).map(|x| (x as f32) * step).collect()
}

fn bench_func(c: &mut Criterion) {
    //
    let array_len = 4;
    let count = 50;

    let data = prepare_data(array_len, count);

    c.bench_function("f32x4_to_srgb8", |b| {
        b.iter(|| {
            for i in 0..count {
                let input: [f32; 4] = data[i * array_len..(i + 1) * array_len].try_into().unwrap();

                black_box(f32x4_to_srgb8(black_box(input)));
            }
        })
    });

    c.bench_function("fast_srgb8", |b| {
        b.iter(|| {
            for &f in data.iter() {
                black_box(f32_to_srgb8(black_box(f)));
            }
        })
    });
}

criterion_group!(benches, bench_func);
criterion_main!(benches);
