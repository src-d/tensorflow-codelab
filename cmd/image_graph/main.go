package main

import (
	"io/ioutil"
	"log"
	"os"

	"github.com/src-d/tensorflow-codelab"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	modelPath := os.Args[1]
	imagePath := os.Args[2]
	graph := tf.NewGraph()
	modelBytes, err := ioutil.ReadFile(modelPath)
	if err != nil {
		log.Fatalf("Reading %s: %v", modelPath, err)
	}
	err = graph.Import(modelBytes, "")
	if err != nil {
		log.Fatalf("Loading the model: %v", err)
	}
	width, height := codelab.GetInputShape(graph)
	log.Printf("Input size: %dx%d\n", width, height)
	image, err := codelab.LoadImage(imagePath, width, height)
	if err != nil {
		log.Fatalf("Reading image: %v", err)
	}
	session, err := tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		log.Fatalf("Opening Tensorflow session: %v", err)
	}
	result, err := codelab.RunInference(graph, session, image)
	if err != nil {
		log.Fatalf("Inference: %v", err)
	}
	filtered := codelab.SelectTopN(result, 5)
	for _, conf := range filtered {
		log.Printf("%-16s %f\n", codelab.ImagenetClasses[conf.Index], conf.Value)
	}
}
