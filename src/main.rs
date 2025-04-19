use crate::mnist::MnistData;
use crate::network::Network;
use ndarray_rand::rand::thread_rng;

mod mnist;
mod network;

fn main() {
    const INPUT_NEURONS: usize = 28 * 28;
    const OUTPUT_NEURONS: usize = 10;

    let mut network = Network::new(vec![INPUT_NEURONS, 30, OUTPUT_NEURONS]);
    let mut mnist_data = MnistData::new().unwrap();
    let mut rng = thread_rng();

    network.stochastic_gradient_descent(&mut mnist_data, 30, 10, 3.0, &mut rng);
}
