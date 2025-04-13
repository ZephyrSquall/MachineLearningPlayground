use ndarray::{Array, Array2, array};
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};

fn main() {
    // Arbitrary network sizes and input activation for testing.
    let network = Network::new(vec![3, 6, 10]);
    let activation = array![[2.0], [4.0], [-0.5]];
    let output = network.feedforward(activation);
    println!("{:6.3}", &output);
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
    fn feedforward(&self, mut activation: Array2<f64>) -> Array2<f64> {
        // For each layer, get the matrices representing its biases and weights.
        for (biases, weights) in self.biases.iter().zip(self.weights.iter()) {
            // Calculate the next state of the activation array (using a_prime = σ(w.a + b)). This
            // line calculates just w.a + b, the argument to σ (the sigmoid function).
            activation = weights.dot(&activation) + biases;
            // Apply the sigmoid function to every element in the activation array to get its actual
            // value.
            activation.map_inplace(|z: &mut f64| *z = sigmoid(*z));
        }
        // The last state of the activation array gives the state of the output layer, so return it.
        activation
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}
