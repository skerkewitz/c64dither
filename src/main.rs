use std::{fs, io, env};
use std::collections::HashSet;
use std::error::Error;
use std::fs::DirEntry;
use std::path::Path;

use image::{ImageBuffer, RgbaImage, RgbImage};
use image::error::DecodingError;
use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use palette::{GammaSrgb, GetHue, Hsl, Hsv, IntoColor, Lab, LinSrgb, RgbHue, Saturate, Srgb};
use rayon::prelude::*;
use simple_error::*;
use vecmath::{vec3_add, vec3_scale};

use std::process::exit;

use lazy_static::lazy_static;

type RGBColor3f = [f32; 3];
type RGBColor3b = [u8; 3];

// const C64_BLACK: RGBColor3b = [0, 0, 0];
// const C64_WHITE: RGBColor3b = [255, 255, 255];
// const C64_RED: RGBColor3b = [104, 55, 43];
// const C64_CYAN: RGBColor3b = [112, 164, 178];
// const C64_PURPLE: RGBColor3b = [111, 61, 134];
// const C64_GREEN: RGBColor3b = [88, 141, 67];
// const C64_BLUE: RGBColor3b = [53, 40, 121];
// const C64_YELLOW: RGBColor3b = [184, 199, 111];
// const C64_ORANGE: RGBColor3b = [111, 79, 37];
// const C64_BROWN: RGBColor3b = [67, 57, 0];
// const C64_LIGHT_RED: RGBColor3b = [154, 103, 89];
// const C64_DARK_GREY: RGBColor3b = [68, 68, 68];
// const C64_GREY: RGBColor3b = [108, 108, 108];
// const C64_LIGHT_GREEN: RGBColor3b = [154, 210, 132];
// const C64_LIGHT_BLUE: RGBColor3b = [108, 94, 181];
// const C64_LIGHT_GREY: RGBColor3b = [149, 149, 149];

const C64_BLACK: RGBColor3b = [0x00, 0x00, 0x00];
const C64_WHITE: RGBColor3b = [0xff, 0xff, 0xff];
const C64_RED: RGBColor3b = [0x9f, 0x4e, 0x44];
const C64_CYAN: RGBColor3b = [0x6a, 0xbf, 0xc6];
const C64_PURPLE: RGBColor3b = [0xa0, 0x57, 0xa3];
const C64_GREEN: RGBColor3b = [0x5c, 0xab, 0x5e];
const C64_BLUE: RGBColor3b = [0x50, 0x45, 0x9b];
const C64_YELLOW: RGBColor3b = [0xc9, 0xd4, 0x87];
const C64_ORANGE: RGBColor3b = [0xa1, 0x68, 0x3c];
const C64_BROWN: RGBColor3b = [0x6d, 0x54, 0x12];
const C64_LIGHT_RED: RGBColor3b = [0xcb, 0x7e, 0x75];
const C64_DARK_GREY: RGBColor3b = [0x62, 0x62, 0x62];
const C64_GREY: RGBColor3b = [0x89, 0x89, 0x89];
const C64_LIGHT_GREEN: RGBColor3b = [154, 210, 132];
const C64_LIGHT_BLUE: RGBColor3b = [0x88, 0x7e, 0xcb];
const C64_LIGHT_GREY: RGBColor3b = [0xad, 0xad, 0xad];


const C64_PALETTE_ALL_3B: [RGBColor3b; 16] = [
    C64_BLACK,          // black
    C64_WHITE,    // white
    C64_RED,        // red
    C64_CYAN,    // cyan
    C64_PURPLE,    // violet / purple
    C64_GREEN,    // green
    C64_BLUE,    // blue
    C64_YELLOW,    // yellow
    C64_ORANGE,    // orange
    C64_BROWN,    // brown
    C64_LIGHT_RED,    // light red
    C64_DARK_GREY,    // dark grey / grey 1
    C64_GREY,    // grey 2
    C64_LIGHT_GREEN,    // light green
    C64_LIGHT_BLUE,    // light blue
    C64_LIGHT_GREY,    // light grey / grey 3
];

lazy_static! { static ref C64_PALETTE_ALL_3F: Vec<(RGBColor3f, RGBColor3b)> = C64_PALETTE_ALL_3B.iter()
        .cloned()
        .map(|i| (rgb_to_vec3f(i), i))
        .collect();
}

fn rgb_to_vec3f(rgb: RGBColor3b) -> RGBColor3f {
    [rgb[0] as f32 / 255.0, rgb[1] as f32 / 255.0, rgb[2] as f32 / 255.0]
}

fn rgb_to_lab3f(rgb: RGBColor3f) -> RGBColor3f {
    let rgb = Srgb::from_components((rgb[0], rgb[1], rgb[2]));
    let lab: Lab = rgb.into_lab();
    return [lab.a, lab.b, lab.l];
}


fn rgbv_error_table(rgb: [f32; 3], error: [f32; 3]) -> Vec<(u32, RGBColor3f, [u8; 3])> {

    let mut errors: Vec<(u32, RGBColor3f, RGBColor3b)> = C64_PALETTE_ALL_3F.iter().map(|(rgb_3f, rgb_3b)| {
        let mut a1 = vec3_add(rgb_to_lab3f(rgb), vec3_scale(error, 0.7));
        a1[0] = clamp(a1[0], 0.0, 100.0);
        a1[1] = clamp(a1[1], -128.0, 127.0);
        a1[2] = clamp(a1[2], -128.0, 127.0);

        let sub = vecmath::vec3_sub(a1, rgb_to_lab3f(*rgb_3f));
        let d = vecmath::vec3_len(sub);

        ((d * 255.0).abs() as u32, sub, *rgb_3b)
    })
        .collect();

    errors.sort_by(|a, b| a.0.cmp(&b.0));

    return errors;
}


pub fn clamp(s: f32, min: f32, max: f32) -> f32 {
    assert!(min <= max);
    let mut x = s;
    if x < min {
        x = min;
    }
    if x > max {
        x = max;
    }
    x
}

fn c64_dither(image: &mut RgbImage) {

    let mut lum_acc_vec_error: RGBColor3f = [0.0, 0.0, 0.0];

    image.enumerate_pixels_mut().for_each(|(x, y, p)| {

        // c64 multi color mode double pixel in x directions
        // XXX SKerkewitz: we should still track the error
        if x % 2 != 0 {
            return
        }

        // reset the error for each line
        if x == 0 {
            lum_acc_vec_error = [0.0, 0.0, 0.0];
        }

        // if (y / 1) % 2 == 0 {
        //     lum_acc_vec_error = [0.0, 0.0, 0.0];
        // }

        let rgbv_error = rgbv_error_table(rgb_to_vec3f(p.0), lum_acc_vec_error);
        let x1 = rgbv_error.first().unwrap();
        p.0 = x1.2;

        // accumulate the error. XXX SKerkewitz: this is actually wrong, but looks fine
        lum_acc_vec_error = vec3_scale(vec3_add(lum_acc_vec_error, x1.1), 0.5);
    });
}

fn c64_multicolor_pixel_fix(image: &mut RgbImage) {
    // Double the x pixel to get 4x8 pixel feel
    for y in 0..image.height() {
        for x in 0..image.width() {
            if x % 2 == 0 {
                continue
            }

            let left = image.get_pixel(x-1, y).clone();
            image.get_pixel_mut(x, y).0 = left.0.clone();
        }
    }
}

/// XXX This does not have a fix background color and should pick the best color fit for replacement
fn c64_reduce_color_per_block(image: &mut RgbImage) {
    // Check for 8x8 region that uses more then 4 color.
    for y in 0..image.height() / 8 {
        for x in 0..image.width() / 8 {

            let offset_x = x * 8;
            let offset_y = y * 8;

            let mut pixel: Vec<(u32, u32, [u8; 3])> = Vec::new();
            for y1 in offset_y..(offset_y + 8) {
                for x1 in offset_x..(offset_x + 8) {
                    let rgb = image.get_pixel(x1, y1).clone();
                    pixel.push((x1, y1, rgb.0));
                }
            }

            let mut color_set: HashSet<[u8; 3]> = pixel.iter()
                .map(|c| c.2)
                .collect();

            if color_set.len() > 4 {
                let mut block: Vec<(usize, [u8; 3], Vec<(u32, u32)>)> = color_set.iter()
                    .map(|c|{
                        let count = pixel.iter()
                            .filter(|x| x.2.eq(c))
                            .map(|x| (x.0, x.1))
                            .collect::<Vec<(u32, u32)>>();
                        (count.len(), c.clone(), count)
                    })
                    .collect();
                fix_pixel_block(&mut block, image);
            }
        }
    }
}

fn stripe_effect(image: &mut RgbImage) {
    for y in 2..(image.height() - 2) {
        for x in 2..(image.width() - 2) {
            if (y / 2) % 2 == 0 {
                continue
            }

            let left = image.get_pixel(x-1, y).clone();
            let right = image.get_pixel(x+1, y).clone();
            if left.0.eq(&right.0) {
                image.get_pixel_mut(x, y).0 = left.0.clone();
            }
        }
    }
}

fn convert_image(in_name: &str, out_name: &str) -> Result<(), Box<dyn Error>> {

    let mut dynamic_image = ImageReader::open(in_name)?.decode()?;
    let mut rgb_image = dynamic_image.as_mut_rgb8().ok_or(SimpleError::new("Could not get mut rgb8"))?;
    c64_dither(rgb_image);
    c64_multicolor_pixel_fix(rgb_image);
    c64_reduce_color_per_block(rgb_image);

    rgb_image.save(out_name)?;
    Ok(())
}

fn fix_pixel_block(block: &mut Vec<(usize, [u8; 3], Vec<(u32, u32)>)>, rgb: &mut RgbImage) {

    while block.len() > 4 {

        block.sort_by(|a, b| a.0.cmp(&b.0).reverse());

        let last = block.remove(block.len() - 1);
        let first = block.first().unwrap();

        for point in last.2 {
            rgb.get_pixel_mut(point.0, point.1).0 = first.1.clone();
        }

        block[0] = (first.0 + last.0, first.1, first.2.clone());
    }
}

fn list_files(input_dir: &Path) -> io::Result<Vec<DirEntry>> {

    let result = fs::read_dir(input_dir)?
        .map(|f| {
            let dir = f?;
            //println!("{:?} {}", dir.path(), dir.file_name().to_str().unwrap());

            if dir.path().is_dir() {
                list_files(&dir.path())
            } else {
                Ok(vec![dir])
            }
        })
        .flatten()
        .flatten()
        .filter(|f|f.file_name().to_str().unwrap().ends_with("jpg"))
        .collect();

    Ok(result)
}


fn dither_single_file(source_path: &Path, out_dir: &Path) {
    let out_file_name = if out_dir.is_dir() {
        let source_file_name = source_path.file_name().unwrap();
        out_dir.join(source_file_name).to_path_buf()
    } else {
        out_dir.to_path_buf()
    };

    fs::create_dir_all(&out_file_name.parent().unwrap()).unwrap();
    let out_name = out_file_name.to_str().unwrap().replace(".jpg", ".png");
    let input_file = source_path.to_str().unwrap();
    match convert_image(input_file, out_name.as_str()) {
        Ok(_) => println!("Did convert '{}' to '{}'...", input_file, out_name),
        Err(e) => eprintln!("Failed to convert '{}' to '{}' because of {}", input_file, out_name, e),
    }
}

fn dither_folder_recursive(source_path: &Path, out_dir: &Path) {
    let vec = list_files(source_path).unwrap();
    vec.into_par_iter().for_each(|dir_entry| {
        let p = dir_entry.path();
        let relative_dir = p.strip_prefix(source_path);
        let input = relative_dir.clone().unwrap();

        let out_path = out_dir.join(input);
        fs::create_dir_all(&out_path.parent().unwrap()).unwrap();

        let input_file = p.to_str().unwrap();
        let output_file = out_path.to_str().unwrap();

        let out_name = output_file.replace(".jpg", ".png");
        match convert_image(input_file, out_name.as_str()) {
            Ok(_) => println!("Did convert '{}' to '{}'...", input_file, out_name),
            Err(e) => eprintln!("Failed to convert '{}' to '{}' because of {}", input_file, out_name, e),
        }
    });
}

fn main() {

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("{} <input dir or file> <output dir>", args[0]);
        exit(0);
    }

    let source_path = Path::new(args[1].as_str());
    let out_dir = Path::new(args[2].as_str());

    if source_path.is_file() {
        dither_single_file(source_path, out_dir);
    } else if source_path.is_dir() {
        dither_folder_recursive(source_path, out_dir);
    } else {
        eprintln!("Given source '{}' is neither file or directory", source_path.to_str().unwrap());
    }
}
