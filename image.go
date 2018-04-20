package codelab

import (
	"github.com/disintegration/imaging"
)

func LoadImage(path string, width, height int) ([][][3]float32, error) {
	img, err := imaging.Open(path)
	if err != nil {
		return nil, err
	}
	adapted := imaging.Fill(img, width, height, imaging.Center, imaging.Lanczos)
	pixels := make([][][3]float32, height)
	for y := 0; y < height; y++ {
		pixels[y] = make([][3]float32, width)
		for x := 0; x < width; x++ {
			r, g, b, _ := adapted.At(x, y).RGBA()
			pixels[y][x][0] = float32(r>>8)/128.0 - 1.0
			pixels[y][x][1] = float32(g>>8)/128.0 - 1.0
			pixels[y][x][2] = float32(b>>8)/128.0 - 1.0
		}
	}
	return pixels, nil
}
