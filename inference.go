package codelab

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func GetInputShape(graph *tf.Graph) (width, height int) {
	input := graph.Operation("module/hub_input/images")
	shape := input.Output(0).Shape()
	return int(shape.Size(1)), int(shape.Size(2))
}

func RunInference(graph *tf.Graph, session *tf.Session, image [][][3]float32) ([]float32, error) {
	input := graph.Operation("module/hub_input/images").Output(0)
	output := graph.Operation("module/MobilenetV2/Logits/output").Output(0)
	images, err := tf.NewTensor([][][][3]float32{image})
	if err != nil {
		return nil, err
	}
	result, err := session.Run(
		map[tf.Output]*tf.Tensor{input: images}, []tf.Output{output}, nil)
	if err != nil {
		return nil, err
	}
	return result[0].Value().([][]float32)[0], nil
}
