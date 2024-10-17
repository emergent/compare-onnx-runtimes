use tract_onnx::prelude::*;

fn main() -> anyhow::Result<()> {
    let model = tract_onnx::onnx()
        .model_for_path("models/mnist.onnx")?
        .with_input_fact(0, f32::fact([1, 1, 28, 28]).into())?
        .incorporate()?
        .into_optimized()?
        .into_runnable()?;

    let input_image = compare_onnx_runtimes::mnist_input_image();

    let input_tensor: Tensor = ndarray::Array4::from_shape_vec((1, 1, 28, 28), input_image)?.into();
    /*
        // below is also ok
        let ndarray_img = ndarray::Array2::from_shape_vec((28, 28), input_image)?;
        let mut input_tensor: Tensor = ndarray_img.into();
        input_tensor.insert_axis(0)?;
        input_tensor.insert_axis(0)?;
    */
    let forward = model.run(tvec!(input_tensor.into()))?;
    let result = forward[0].to_array_view::<f32>()?;
    println!("{:?}", result);

    let mut argmax = f32::MIN;
    let mut ans = 0;
    for (i, x) in result.into_iter().enumerate() {
        println!("{}: {:?}", i, x);

        if *x > argmax {
            argmax = *x;
            ans = i;
        }
    }
    println!("answer is {}", ans);

    Ok(())
}
