package main

import (
	// #include "add.h"
	// #cgo LDFLAGS: -L. -ladd
	"C"
	"net/http"

	"github.com/gin-gonic/gin"
)

type addRequest struct {
	A int `json:"a"`
	B int `json:"b"`
}

type addResponse struct {
	Result int `json:"result"`
}

func add(c *gin.Context) {
	var req addRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	}

	sum := C.add(C.int(req.A), C.int(req.B))
	result := addResponse{Result: int(sum)}
	c.IndentedJSON(http.StatusOK, result)
}

func main() {
	router := gin.Default()
	router.POST("/add", add)
	router.Run("localhost:8080")
}
