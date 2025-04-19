use flate2::read::GzDecoder;
use itertools::Itertools;
use ndarray::{Array, Array2};
use std::{fs::File, io::Read};

// A single pair of a handwritten digit and its correct label. The expected_output is in the format
// of a [10 x 1] array with the neuron linked to the correct value set to 1.0 and all other neurons
// set to 0.0, which is the format of the perfect activation output of the network.
pub struct MnistTrainingDatum {
    pub input: Array2<f64>,
    pub expected_output: Array2<f64>,
}
impl From<MnistTestDatum> for MnistTrainingDatum {
    fn from(value: MnistTestDatum) -> Self {
        // Turn the label into a [10 x 1] array where every value is 0.0 except for the value in the
        // position corresponding to the label, whose value is 1.0.
        let expected_output = Array::from_shape_fn([10, 1], |(i, _j)| {
            if i == value.expected_answer.into() {
                1.0
            } else {
                0.0
            }
        });
        MnistTrainingDatum {
            input: value.input,
            expected_output,
        }
    }
}

// A single pair of a handwritten digit and its correct label. The expected_answer is simply the
// correct value as a simple byte.
pub struct MnistTestDatum {
    pub input: Array2<f64>,
    pub expected_answer: u8,
}

pub struct MnistData {
    pub training_data: Vec<MnistTrainingDatum>,
    pub validation_data: Vec<MnistTestDatum>,
    pub test_data: Vec<MnistTestDatum>,
}

impl MnistData {
    pub fn new() -> MnistData {
        // Take a file path to MNIST data, and returns a Vec of u32s representing each value in the
        // underlying file.
        fn read_bytes(path: &str) -> Vec<u8> {
            let file = File::open(path).unwrap();
            let mut unzipped_file = GzDecoder::new(file);
            let mut bytes = Vec::new();
            unzipped_file.read_to_end(&mut bytes);
            bytes
        }

        // Take the raw bytes of MNIST image data and MNIST label data, and convert these into
        // easy-to-use Rust structs.
        fn images_and_labels_to_data(
            image_bytes: Vec<u8>,
            label_bytes: Vec<u8>,
        ) -> Vec<MnistTestDatum> {
            let mut image_bytes_iter = image_bytes.into_iter();
            let mut label_bytes_iter = label_bytes.into_iter();

            // Read the headers of the image bytes. The headers are four 32-bit integers (so 4
            // headers * 4 bytes each = 16 bytes for the headers), which represent in order: the
            // magic number (2051), the number of images, the number of rows per image, and the
            // number of columns per image. Take the overall iterator of image bytes by reference so
            // it can be reused for the individual image bytes later. The header bytes must be
            // collected so they can be chunked together and read as a single u32.
            let image_headers_bytes = image_bytes_iter.by_ref().take(16).collect::<Vec<_>>();
            let mut image_headers_iter = image_headers_bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_be_bytes(chunk.try_into().unwrap()));

            // The magic number for an image file is 2051. If any other number is found here, either
            // an incorrect file has been provided, or something has gone horribly wrong.
            assert!(image_headers_iter.next() == Some(2051));
            let images = image_headers_iter.next().unwrap();
            let rows = image_headers_iter.next().unwrap();
            let columns = image_headers_iter.next().unwrap();

            // Read the headers of the label bytes. These headers are just two 32-bit integers (8
            // bytes for the headers), which represent in order: the magic number (2049), and the
            // number of labels.
            let label_headers_bytes = label_bytes_iter.by_ref().take(8).collect::<Vec<_>>();
            let mut label_headers_iter = label_headers_bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_be_bytes(chunk.try_into().unwrap()));

            assert!(label_headers_iter.next() == Some(2049));
            let labels = label_headers_iter.next().unwrap();

            // There should be an equal number of images and labels. If not, either an incorrect
            // file has been provided, or something has gone horribly wrong.
            assert!(images == labels);

            // Initialize the MNIST data array that will hold the output of this function.
            let mut data = Vec::with_capacity(images.try_into().unwrap());

            // The image bytes iterator and label bytes iterator have both had their headers
            // consumed, but none of the data bytes, so reuse them for the rest of the data. First,
            // create a new iterator from the image bytes iterator that returns rows*columns chunks
            // of bytes (assuming 28 rows by 28 columns, these will be 784-byte chunks). Then zip
            // this new iterator with the label bytes iterator, to iterate over every pair of an
            // image and its label.
            for (image_chunk, expected_answer) in image_bytes_iter
                .chunks((rows * columns).try_into().unwrap())
                .into_iter()
                .zip(label_bytes_iter)
            {
                // Turn this chunk (i.e. iterator of 784 bytes) into a [784 x 1] array. Map each
                // byte into an array (in this specific case, the standard Rust library kind of
                // array) of one element, to indicate each row in the resulting 2D array has only
                // one element. Then collect them into a vector that can be used to convert into a
                // 2D array.
                let image_input = image_chunk
                    .map(|value| [value as f64 / 255.0])
                    .collect::<Vec<_>>()
                    .into();

                // Use these two arrays to create a MNIST datum and push it to the MNIST data
                // vector.
                data.push(MnistTestDatum {
                    input: image_input,
                    expected_answer,
                })
            }

            data
        }

        let training_image_bytes = read_bytes("data/train-images-idx3-ubyte.gz");
        let training_label_bytes = read_bytes("data/train-labels-idx1-ubyte.gz");
        let mut training_data =
            images_and_labels_to_data(training_image_bytes, training_label_bytes);

        // Create the validation data set by taking the final 10000 training data samples.
        let validation_data = training_data.split_off(50000);

        // Convert the remaining 50000 training data samples from Vec<MnistTestDatum> into
        // Vec<MnistTrainingDatum>.
        let training_data = training_data
            .into_iter()
            .map(MnistTrainingDatum::from)
            .collect();

        let test_image_bytes = read_bytes("data/t10k-images-idx3-ubyte.gz");
        let test_label_bytes = read_bytes("data/t10k-labels-idx1-ubyte.gz");
        let test_data = images_and_labels_to_data(test_image_bytes, test_label_bytes);

        MnistData {
            training_data,
            validation_data,
            test_data,
        }
    }
}

// Debug function to quickly print a simple ASCII representation of a MNIST datum and its associated
// label.
pub fn visualize(mnist_training_datum: &MnistTrainingDatum) {
    const ROWS: usize = 28;
    const COLUMNS: usize = 28;

    for (index, input_activation) in mnist_training_datum.input.iter().enumerate() {
        if index % COLUMNS == 0 {
            println!()
        }

        match input_activation {
            a if *a < 0.2 => {
                print!(" ")
            }
            a if *a < 0.4 => {
                print!("░")
            }
            a if *a < 0.6 => {
                print!("▒")
            }
            a if *a < 0.8 => {
                print!("▓")
            }
            _ => {
                print!("█")
            }
        }
    }

    for (index, output_activation) in mnist_training_datum.expected_output.iter().enumerate() {
        if *output_activation > 0.5 {
            println!("\nAnswer: {index}");
            break;
        }
    }
}
