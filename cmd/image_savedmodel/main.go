package main

import (
	"fmt"
	"log"
	"os"

	"github.com/src-d/tensorflow-codelab"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	modelPath := os.Args[1]
	imagePath := os.Args[2]
	savedModel, err := tf.LoadSavedModel(modelPath, []string{"train"}, nil)
	if err != nil {
		log.Fatalf("Loading the model: %v", err)
	}
	width, height := codelab.GetInputShape(savedModel.Graph)
	image, err := codelab.LoadImage(imagePath, width, height)
	if err != nil {
		log.Fatalf("Reading image: %v", err)
	}
	result, err := codelab.RunInference(savedModel.Graph, savedModel.Session, image)
	if err != nil {
		log.Fatalf("Inference: %v", err)
	}
	filtered := codelab.SelectTopN(result, 5)
	for _, conf := range filtered {
		fmt.Printf("%-16s %f\n", codelab.ImagenetClasses[conf.Index], conf.Value)
	}
}
