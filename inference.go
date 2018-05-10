package codelab

import (
	"errors"
	"log"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// GetInputShape finds the input image dimensions.
func GetInputShape(graph *tf.Graph) (width, height int) {
	input := graph.Operation("module/hub_input/images")
	if input == nil {
		log.Fatal("Cannot find tensor \"module/hub_input/images\"")
	}
	shape := input.Output(0).Shape()
	return int(shape.Size(1)), int(shape.Size(2))
}

// RunInference executes the model and returns the logits.
func RunInference(graph *tf.Graph, session *tf.Session, image [][][3]float32) ([]float32, error) {
	inputOp := graph.Operation("module/hub_input/images")
	if inputOp == nil {
		return nil, errors.New("Cannot find tensor \"module/hub_input/images\"")
	}
	input := inputOp.Output(0)
	outputOp := graph.Operation("module/MobilenetV2/Logits/output")
	if outputOp == nil {
		return nil, errors.New("Cannot find tensor \"module/MobilenetV2/Logits/output\"")
	}
	output := outputOp.Output(0)
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
