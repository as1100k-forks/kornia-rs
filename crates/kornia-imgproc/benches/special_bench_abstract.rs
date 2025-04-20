use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use image::{ImageBuffer, Rgb, RgbImage};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::{features::{dog_response, HarrisResponse}, flip::horizontal_flip};
use rand::Rng;
use rayon::slice::ParallelSliceMut;


// TODO: we can also test between u8 vs f32 and RGB vs greyscale

fn generate_random_rgb_image_u8(width: usize, height: usize) -> (RgbImage, Image<u8, 3>) {
    let mut img = RgbImage::new(width as u32, height as u32);
    let mut rng = rand::thread_rng();

    let mut kornia_img = Image::from_size_val(ImageSize { width, height }, 0u8).unwrap();

    for (image_img, kornia_img) in img.as_parallel_slice_mut().iter_mut().zip(kornia_img.as_slice_mut()) {
        let c = rng.gen::<u8>();
        *image_img = c;
        *kornia_img = c;
    }

    (img, kornia_img)
}

fn generate_random_rgb_image_f32(width: usize, height: usize) -> (RgbImage, Image<f32, 3>) {
    let mut img = RgbImage::new(width as u32, height as u32);
    let mut rng = rand::thread_rng();

    let mut kornia_img = Image::from_size_val(ImageSize { width, height }, 0.0f32).unwrap();

    for (image_img, kornia_img) in img.as_parallel_slice_mut().iter_mut().zip(kornia_img.as_slice_mut()) {
        let c = rng.gen::<u8>();
        *image_img = c;
        *kornia_img = (c as f32) / 255.0;
    }

    (img, kornia_img)
}

// make sure this benchmark file is shown on Cargo.toml
// cd crates/kornia-imgproc
// cargo bench --bench special_bench_abstract
fn bench_images(c: &mut Criterion) {
    let mut group = c.benchmark_group("Special/ImageComparison");
    let mut rng = rand::thread_rng();

    for (width, height) in [
        ( 854,  480),  // 480p
        (1280,  720),  // 720p
        (1920, 1080),  // 1080p
        (2560, 1440),  // 1440p
        (3840, 2160),  // 4k
    ].into_iter() {
        group.throughput(criterion::Throughput::Elements((width * height) as u64));

        let parameter_string = format!("{}x{}", width, height);


        // horizontal flip
        {
            let (image, kornia) = generate_random_rgb_image_u8(width, height);
            let mut image_output = ImageBuffer::<Rgb<u8>, _>::new(width as u32, height as u32);
            let mut kornia_output = Image::from_size_val(ImageSize { width, height }, 0u8).unwrap();
            group.bench_with_input(
                BenchmarkId::new("krn/flip", &parameter_string),
                &(&kornia, &mut kornia_output),
                |b, i| {
                    let (krn_inp, mut krn_dst) = (i.0, i.1.clone());
                    b.iter(|| black_box(kornia_imgproc::flip::horizontal_flip(krn_inp, &mut krn_dst)))
                },
            );
            group.bench_with_input(
                BenchmarkId::new("img/flip", &parameter_string),
                &(&image, &mut image_output),
                |b, i| {
                    let (img_inp, mut img_dst) = (i.0, i.1.clone());
                    b.iter(|| black_box(image::imageops::flip_horizontal_in(img_inp, &mut img_dst)))
                },
            );
        }

        // resize is f32 for kornia and u8 for image (crate)
        // resize (over sampling) to 4000x4000 using Nearest
        {
            let (image, kornia) = generate_random_rgb_image_f32(width, height);
            let mut image_output = ImageBuffer::<Rgb<u8>, _>::new(width as u32, height as u32);
            let mut kornia_output_resize = Image::from_size_val(ImageSize { width: 4000, height: 4000 }, 0.0f32).unwrap();
            group.bench_with_input(
                BenchmarkId::new("krn/resize_oversample", &parameter_string),
                &(&kornia, &mut kornia_output_resize),
                |b, i| {
                    let (krn_inp, mut krn_dst) = (i.0, i.1.clone());
                    b.iter(|| black_box(kornia_imgproc::resize::resize_native(krn_inp, &mut krn_dst, kornia_imgproc::interpolation::InterpolationMode::Nearest)))
                },
            );
            group.bench_with_input(
                BenchmarkId::new("img/resize_oversample", &parameter_string),
                &(&image, &mut image_output),
                |b, i| {
                    let (img_inp, mut _img_dst) = (i.0, i.1.clone());
                    b.iter(|| black_box(image::imageops::resize(img_inp, 4000, 4000, image::imageops::FilterType::Nearest)))
                },
            );
        }

        // warp perspective (image crate does not have it)

        // resize (under sampling) to 400x400 using Nearest
        {
            let (image, kornia) = generate_random_rgb_image_f32(width, height);
            let mut image_output = ImageBuffer::<Rgb<u8>, _>::new(width as u32, height as u32);
            let mut kornia_output_resize = Image::from_size_val(ImageSize { width: 400, height: 400 }, 0.0f32).unwrap();
            group.bench_with_input(
                BenchmarkId::new("krn/resize_undersample", &parameter_string),
                &(&kornia, &mut kornia_output_resize),
                |b, i| {
                    let (krn_inp, mut krn_dst) = (i.0, i.1.clone());
                    b.iter(|| black_box(kornia_imgproc::resize::resize_native(krn_inp, &mut krn_dst, kornia_imgproc::interpolation::InterpolationMode::Nearest)))
                },
            );
            group.bench_with_input(
                BenchmarkId::new("img/resize_undersample", &parameter_string),
                &(&image, &mut image_output),
                |b, i| {
                    let (img_inp, mut _img_dst) = (i.0, i.1.clone());
                    b.iter(|| black_box(image::imageops::resize(img_inp, 400, 400, image::imageops::FilterType::Nearest)))
                },
            );
        }

        // color conversion
        {
            let (image, kornia) = generate_random_rgb_image_u8(width, height);
            let mut image_output = ImageBuffer::<Rgb<u8>, _>::new(width as u32, height as u32);
            let mut kornia_output = Image::from_size_val(ImageSize { width, height }, 0u8).unwrap();
            group.bench_with_input(
                BenchmarkId::new("krn/rgb2gray", &parameter_string),
                &(&kornia, &mut kornia_output),
                |b, i| {
                    let (krn_inp, mut krn_dst) = (i.0, i.1.clone());
                    b.iter(|| black_box(kornia_imgproc::color::gray_from_rgb_u8(krn_inp, &mut krn_dst)))
                },
            );
            group.bench_with_input(
                BenchmarkId::new("img/rgb2gray", &parameter_string),
                &(&image, &mut image_output),
                |b, i| {
                    let (img_inp, mut img_dst) = (i.0, i.1.clone());
                    b.iter(|| black_box(image::imageops::grayscale(img_inp)))
                },
            );
        }
    }
    group.finish();
}

// fn bench_harris_response(c: &mut Criterion) {
//     let mut group = c.benchmark_group("Features");
//     let mut rng = rand::thread_rng();

//     for (width, height) in [(1920, 1080)].iter() {
//         group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

//         let parameter_string = format!("{}x{}", width, height);

//         // input image
//         let image_data: Vec<f32> = (0..(*width * *height))
//             .map(|_| rng.gen_range(0.0..1.0))
//             .collect();
//         let image_size = [*width, *height].into();

//         let image_f32: Image<f32, 1> = Image::new(image_size, image_data).unwrap();

//         // output image
//         let response_f32: Image<f32, 1> = Image::from_size_val(image_size, 0.0).unwrap();
//         let mut harris_response = HarrisResponse::new(image_size);

//         group.bench_with_input(
//             BenchmarkId::new("harris", &parameter_string),
//             &(&image_f32, &response_f32),
//             |b, i| {
//                 let (src, mut dst) = (i.0, i.1.clone());
//                 b.iter(|| black_box(harris_response.compute(src, &mut dst)))
//             },
//         );
//     }
//     group.finish();
// }

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(std::time::Duration::new(10, 0));
    targets = bench_images
);
criterion_main!(benches);


#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use rayon::slice::ParallelSlice;

    #[test]
    fn test_save() {
        const WIDTH: usize = 1920;
        const HEIGHT: usize = 1080;
        let (image, kornia) = generate_random_rgb_image_u8(WIDTH, HEIGHT);

        assert_eq!(kornia.as_slice(), image.as_parallel_slice());

        image.save("./test_image.png").unwrap();
        ImageBuffer::<Rgb<u8>, _>::from_raw(WIDTH as u32, HEIGHT as u32, kornia.as_slice()).unwrap().save("./test_kornia.png").unwrap();

    }
}
