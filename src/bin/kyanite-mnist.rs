use compare_onnx_runtimes::argmax;
use kn_graph::{
    cpu::cpu_eval_graph, dot::graph_to_svg, dtype::DTensor, ndarray::Array4,
    onnx::load_graph_from_onnx_path, optimizer::optimize_graph,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // 1. create tensor
    let input_image = compare_onnx_runtimes::mnist_input_image();
    let input_tensor = Array4::from_shape_vec((1, 1, 28, 28), input_image)?
        .into_dyn()
        .into_shared();
    let inputs = [DTensor::F32(input_tensor)];

    // 2. load model and initialize session
    let graph = load_graph_from_onnx_path("models/mnist.onnx", false)?;
    let graph = optimize_graph(&graph, Default::default());
    graph_to_svg("mnist.svg", &graph, false, false)?;

    // 3. run inference
    let start = Instant::now();
    let outputs: Vec<DTensor> = cpu_eval_graph(&graph, 8, &inputs);
    let end = start.elapsed();
    println!("{:?}", outputs);

    // 4. fetch result
    let ans = argmax(outputs[0].unwrap_f32().unwrap().iter()).unwrap();

    println!("answer is {}", ans);
    println!("time: {}.{:09}", end.as_secs(), end.subsec_nanos());

    Ok(())
}
