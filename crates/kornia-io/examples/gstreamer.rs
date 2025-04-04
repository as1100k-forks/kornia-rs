use kornia_image::{Image, ImageSize};
use kornia_io::jpeg::write_image_jpeg_rgb8;
use reqwest::blocking::get;
use std::io::copy;
use std::path::PathBuf;
use std::{fs::File, time::Duration};
use tempfile::{tempdir, TempDir};

const FILE_NAME: &str = "video.mp4";
const VIDEO_LINK: &str =
    "https://github.com/kornia/tutorials/raw/refs/heads/master/data/sharpening.mp4";

fn download_video<'a>() -> (PathBuf, TempDir) {
    let response = get(VIDEO_LINK).expect("Failed to download video");
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let temp_file_path = temp_dir.path().join(FILE_NAME);
    let mut temp_file = File::create(&temp_file_path).expect("Failed to create temp file");

    copy(
        &mut response.bytes().expect("Failed to read response").as_ref(),
        &mut temp_file,
    )
    .expect("Failed to write video to temp file");

    println!("Video downloaded to: {:?}", temp_file_path);
    (temp_file_path, temp_dir)
}

fn main() {
    let (video_path, _temp_dir) = download_video();

    let pipeline_desc = format!(
        "filesrc location={} ! decodebin ! videoconvert ! video/x-raw,format=RGB,width=1024,height=688,framerate=8/1 ! appsink name=sink sync=false",
        video_path.to_str().unwrap()
    );

    let mut stream_capture = kornia_io::stream::StreamCapture::new(&pipeline_desc).unwrap();
    stream_capture.start().unwrap();
    std::thread::sleep(Duration::from_secs(1));

    for i in 0..5 {
        let img = stream_capture
            .grab()
            .expect("Failed to grab the image")
            .unwrap();

        let img_size = img.0.shape;

        let img_clone = img.0.clone();
        let img_new = Image::new(
            ImageSize {
                width: img_size[0],
                height: img_size[1],
            },
            img_clone.into_vec(),
        )
        .unwrap();

        write_image_jpeg_rgb8(format!("./tests/data/video-{}.jpeg", i), &img_new).unwrap();
        std::thread::sleep(Duration::from_secs(1));
    }
}
