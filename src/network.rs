use crate::mnist::{MnistData, MnistTrainingDatum};
use ndarray::{Array, Array2, Axis, concatenate};
use ndarray_rand::{
    RandomExt,
    rand::{Rng, seq::SliceRandom},
    rand_distr::StandardNormal,
};

pub struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Network {
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
    pub fn feedforward(&self, input_activation: &Array2<f64>) -> Array2<f64> {
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
    pub fn stochastic_gradient_descent<R: Rng + ?Sized>(
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
        // Combine each training datum input and expected output into a single matrix where each
        // column corresponds to a separate datum.
        let training_input_matrix = concatenate(
            Axis(1),
            &mini_batch
                .iter()
                .map(|mnist_training_datum| mnist_training_datum.input.view())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let training_expected_output_matrix = concatenate(
            Axis(1),
            &mini_batch
                .iter()
                .map(|mnist_training_datum| mnist_training_datum.expected_output.view())
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let (mut nabla_biases, mut nabla_weights) = self.backpropagate(
            &training_input_matrix,
            &training_expected_output_matrix,
            mini_batch.len(),
        );

        for (bias, nabla_bias) in self.biases.iter_mut().zip(nabla_biases.iter_mut()) {
            // To avoid another allocation, the bias gradient can be safely mapped in-place to scale
            // it according to the learning rate. It will be dropped after this calculation due to
            // having no further uses, so its state after this calculation doesn't matter.
            nabla_bias.mapv_inplace(|nb| nb * learning_rate / mini_batch.len() as f64);
            // The signature of the AddAssign operation for Arrays requires the right operand to be
            // a (non-mutable) reference, so deference the mutable reference and reborrow as a
            // (non-mutable) reference.
            *bias -= &*nabla_bias;
        }
        for (weight, nabla_weight) in self.weights.iter_mut().zip(nabla_weights.iter_mut()) {
            nabla_weight.mapv_inplace(|nb| nb * learning_rate / mini_batch.len() as f64);
            *weight -= &*nabla_weight;
        }
    }

    // Calculate the deltas of all biases and weights.
    fn backpropagate(
        &self,
        training_input_matrix: &Array2<f64>,
        training_expected_output_matrix: &Array2<f64>,
        mini_batch_size: usize,
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

        let mut activation = training_input_matrix.clone();
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
        let mut delta = (&activation - training_expected_output_matrix) * &*z;
        activations.push(activation);

        // The index of the final activation layer is self.num_layers - 1. However, the second-last
        // index is needed, so which works out to self.num_layers - 2
        // Note that delta is a matrix where every column is the delta for the corresponding input
        // in the training input matrix. nabla_biases wants the sum of all deltas, which means
        // summing up each row in delta. This can be achieved with the dot product with a column
        // matrix of all ones. Similarly, nabla_weights wants the sum of the dot product of the
        // delta with each activation column vector, but it turns out combining all the activation
        // column vectors into a single matrix causes the dot product of the delta with it to
        // automatically sum everything up for us.
        nabla_biases[self.num_layers - 2] = delta.clone().dot(&Array2::ones((mini_batch_size, 1)));
        nabla_weights[self.num_layers - 2] = delta.dot(&activations[self.num_layers - 2].t());

        for l in 2..self.num_layers {
            let z = &mut zs[self.num_layers - 1 - l];
            z.mapv_inplace(sigmoid_derivative);
            delta = self.weights[self.num_layers - l].t().dot(&delta) * &*z;

            nabla_biases[self.num_layers - 1 - l] =
                delta.clone().dot(&Array2::ones((mini_batch_size, 1)));
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
