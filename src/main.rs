use flate2::read::GzDecoder;
use itertools::Itertools;
use ndarray::{Array, Array2};
use ndarray_rand::{
    RandomExt,
    rand::{Rng, seq::SliceRandom, thread_rng},
    rand_distr::StandardNormal,
};
use std::{fs::File, io::Read};

fn main() {
    const INPUT_NEURONS: usize = 28 * 28;
    const OUTPUT_NEURONS: usize = 10;

    let mut network = Network::new(vec![INPUT_NEURONS, 30, OUTPUT_NEURONS]);
    let mut mnist_data = MnistData::new();
    let mut rng = thread_rng();

    network.stochastic_gradient_descent(&mut mnist_data, 30, 10, 3.0, &mut rng);
}

struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    fn new(sizes: Vec<usize>) -> Network {
        Network {
            num_layers: sizes.len(),
            biases: sizes
                // For each size in sizes...
                .iter()
                // Except the first one...
                .skip(1)
                // Make a [size x 1] array of numbers randomly chosen from a standard normal
                // distribution (a normal distribution with mean 0 and standard deviation 1)...
                .map(|&size| Array::random((size, 1), StandardNormal))
                // And collect each of these arrays into a Vec.
                .collect(),
            weights: sizes
                // For each size in sizes...
                .iter()
                // Paired with the following size in sizes...
                .zip(sizes.iter().skip(1))
                // Make a [next_size x current_size] array of numbers randomly chosen from a
                // standard normal distribution...
                .map(|(&current_size, &next_size)| {
                    Array::random((next_size, current_size), StandardNormal)
                })
                // And collect each of these arrays into a Vec.
                .collect(),
            sizes,
        }
    }

    // Calculates the activations of the output layer, given the activations of the input layer. The
    // input activation must be a [self.sizes[0] x 1] array (i.e. it must match the size of the
    // input layer), or this method will panic due to improper dimensions on the dot product.
    fn feedforward(&self, input_activation: &Array2<f64>) -> Array2<f64> {
        // Get the matrices representing the biases and weights for the second layer (the first
        // layer after the input layer) and compute its activation matrix (using
        // a_prime = σ(w.a + b)). First w.a + b is calculated, then the sigmoid function σ(z) is
        // applied to each element.
        let mut activation = self.weights[0].dot(input_activation) + &self.biases[0];
        activation.map_inplace(|z: &mut f64| *z = sigmoid(*z));

        // Repeat this process for all remaining layers, overwriting the activation matrix each time
        // as only the prior activation matrix is needed.
        for (biases, weights) in self.biases.iter().zip(self.weights.iter()).skip(1) {
            activation = weights.dot(&activation) + biases;
            activation.map_inplace(|z: &mut f64| *z = sigmoid(*z));
        }

        // The last state of the activation array gives the state of the output layer, so return it.
        activation
    }

    // Using stochastic gradient descent, trains the network. The MNIST training data is shuffled,
    // then divided into batches of size mini_batch_size. For each batch, the overall gradient of
    // the cost function is calculated through backpropagation, and the network's biases and weights
    // are adjusted accordingly. After all batches have been used, the network is evaluated by
    // running it against the test data and printing how many digits were correctly classified. Then
    // this entire process is repeated for the given number of epochs.
    fn stochastic_gradient_descent<R: Rng + ?Sized>(
        &mut self,
        mnist_data: &mut MnistData,
        epochs: u32,
        mini_batch_size: usize,
        learning_rate: f64,
        rng: &mut R,
    ) {
        for epoch in 0..epochs {
            mnist_data.training_data.shuffle(rng);

            // Train on the training data.
            for mini_batch in mnist_data.training_data.chunks(mini_batch_size) {
                self.update_mini_batch(mini_batch, learning_rate);
            }

            // Use the test data to determine accuracy.
            self.evaluate(epoch, mnist_data);
        }
    }

    // Adjust the network's biases and weights according to the given batch of training data.
    fn update_mini_batch(&mut self, mini_batch: &[MnistTrainingDatum], learning_rate: f64) {
        // Get zero-initialized matrices matching the size of the weights and biases matrices.
        let mut nabla_biases: Vec<Array2<f64>> = self
            .biases
            .iter()
            .map(|bias| Array::zeros(bias.raw_dim()))
            .collect();
        let mut nabla_weights: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|weight| Array::zeros(weight.raw_dim()))
            .collect();

        for mnist_training_datum in mini_batch {
            let (delta_nabla_biases, delta_nabla_weights) =
                self.backpropagate(mnist_training_datum);

            // Add each change in bias gradient to the corresponding bias gradient
            for (nabla_bias, delta_nabla_bias) in
                nabla_biases.iter_mut().zip(delta_nabla_biases.iter())
            {
                *nabla_bias += delta_nabla_bias;
            }
            // Add each change in weight gradient to the corresponding weight gradient
            for (nabla_weight, delta_nabla_weight) in
                nabla_weights.iter_mut().zip(delta_nabla_weights.iter())
            {
                *nabla_weight += delta_nabla_weight;
            }
        }

        for (bias, nabla_bias) in self.biases.iter_mut().zip(nabla_biases.iter_mut()) {
            // To avoid another allocation, the bias gradient can be safely mapped in-place to scale
            // it according to the learning rate. It will be dropped after this calculation due to
            // having no further uses, so its state after this calculation doesn't matter.
            nabla_bias.mapv_inplace(|nb| nb * learning_rate / mini_batch.len() as f64);
            // The signature of the AddAssign operation for Arrays requires the right operand to be
            // a (non-mutable) reference, so deference the mutable reference and reborrow as a
            // (non-mutable) reference.
            //println!("Bias before: {:6.3}", bias);
            *bias -= &*nabla_bias;
            //println!("Bias after: {:6.3}", bias);
        }
        for (weight, nabla_weight) in self.weights.iter_mut().zip(nabla_weights.iter_mut()) {
            nabla_weight.mapv_inplace(|nb| nb * learning_rate / mini_batch.len() as f64);
            *weight -= &*nabla_weight;
        }
    }

    // Calculate the deltas of all biases and weights.
    fn backpropagate(
        &self,
        mnist_training_datum: &MnistTrainingDatum,
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        // Get zero-initialized matrices matching the size of the weights and biases matrices.
        let mut nabla_biases: Vec<Array2<f64>> = self
            .biases
            .iter()
            .map(|bias| Array::zeros(bias.raw_dim()))
            .collect();
        let mut nabla_weights: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|weight| Array::zeros(weight.raw_dim()))
            .collect();

        let mut activation = mnist_training_datum.input.clone();
        let mut activations = Vec::with_capacity(self.num_layers);
        let mut zs = Vec::with_capacity(self.num_layers);

        // Pushing arrays to a vector moves the array, which then prevents references to that array
        // from being reused. Compared to the book, the order of operations in this loop had to be
        // carefully shifted to comply with Rust's ownership rules. In particular, rather than
        // pushing the first activation prior to the loop and pushing the last activation during the
        // loop, things have been rearranged so the first activation is pushed during the loop and
        // the last activation is pushed after the loop.
        for (bias, weight) in self.biases.iter().zip(self.weights.iter()) {
            let z = weight.dot(&activation) + bias;
            activations.push(activation);
            activation = z.mapv(sigmoid);
            zs.push(z);
        }

        // For the following lines, needing to access the last element of a mutably-borrowed vector
        // is a comment requirement. However, the length of a vector cannot be read while mutably
        // borrowing it, as reading the length of the vector requires an additional immutable
        // borrow, which is not allowed by Rust's borrowing rules. To get around this,
        // self.num_layers is used instead, as the number of elements in all of these vectors
        // depends on how many layers the network has. In particular:
        //  - zs has (self.num_layers - 1) elements
        //  - activations has (self.num_layers) elements
        //  - nabla_biases has (self.num_layers - 1) elements
        //  - nabla_weights has (self.num_layers - 1) elements
        // Note that the index of the final element is 1 less than how many elements it has.

        let z = &mut zs[self.num_layers - 2];
        z.mapv_inplace(sigmoid_derivative);
        let mut delta = (&activation - &mnist_training_datum.expected_output) * &*z;
        // delta.mapv_inplace(sigmoid_derivative);
        activations.push(activation);

        // The index of the final activation layer is self.num_layers - 1. However, the second-last
        // index is needed, so which works out to self.num_layers - 2
        nabla_biases[self.num_layers - 2] = delta.clone();
        nabla_weights[self.num_layers - 2] = delta.dot(&activations[self.num_layers - 2].t());

        for l in 2..self.num_layers {
            let z = &mut zs[self.num_layers - 1 - l];
            z.mapv_inplace(sigmoid_derivative);
            delta = self.weights[self.num_layers - l].t().dot(&delta) * &*z;

            nabla_biases[self.num_layers - 1 - l] = delta.clone();
            nabla_weights[self.num_layers - 1 - l] =
                delta.dot(&activations[self.num_layers - 1 - l].t());
        }

        (nabla_biases, nabla_weights)
    }

    // Run the network on all test data and report how many digits were correctly identified.
    fn evaluate(&self, epoch: u32, mnist_data: &MnistData) {
        let correct_answers = mnist_data
            // For each MNIST test datum...
            .test_data
            .iter()
            // Filter down to only the datums that this network gives the correct response to.
            .filter(|mnist_test_datum| {
                self.feedforward(&mnist_test_datum.input)
                    // Network::feedforward returns a [10 x 1] array of activations. Find the
                    // activation with the highest value...
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    // Get the index of that neuron...
                    .map(|(index, _)| index)
                    .expect("All network outputs should be an array with at least 1 element (specifically 10)")
                    // And check if the index matches the correct answer.
                    == mnist_test_datum.expected_answer as usize
            })
            // Finally, count the correct responses.
            .count();

        println!(
            "Epoch {epoch}: {correct_answers} / {}",
            mnist_data.test_data.len()
        );
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}

fn sigmoid_derivative(z: f64) -> f64 {
    (1.0 - sigmoid(z)) * sigmoid(z)
}

// A single pair of a handwritten digit and its correct label. The expected_output is in the format
// of a [10 x 1] array with the neuron linked to the correct value set to 1.0 and all other neurons
// set to 0.0, which is the format of the perfect activation output of the network.
struct MnistTrainingDatum {
    input: Array2<f64>,
    expected_output: Array2<f64>,
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
struct MnistTestDatum {
    input: Array2<f64>,
    expected_answer: u8,
}

struct MnistData {
    training_data: Vec<MnistTrainingDatum>,
    validation_data: Vec<MnistTestDatum>,
    test_data: Vec<MnistTestDatum>,
}

impl MnistData {
    fn new() -> MnistData {
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
fn visualize(mnist_training_datum: &MnistTrainingDatum) {
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
